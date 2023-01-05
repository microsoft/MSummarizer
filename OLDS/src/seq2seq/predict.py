import torch
import nltk
import numpy as np
from tqdm.auto import tqdm
from others.dataset import get_datasets
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


def predict(args, logger, accelerator, tokenizer=None, model=None, dataset=None):

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    if model is None:
        accelerator.free_memory()

        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

        model.resize_token_embeddings(len(tokenizer))
        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        # Prepare model with our `accelerator`.
        model = accelerator.prepare(model)

    if dataset is None:
        dataset = get_datasets(args, accelerator, tokenizer)['test']

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    eval_dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Prepare everything with our `accelerator`.
    eval_dataloader = accelerator.prepare(eval_dataloader)

    def postprocess_text(texts):
        texts = [t.strip() for t in texts]

        # rougeLSum expects newline after each sentence
        texts = ["\n".join(nltk.sent_tokenize(t)) for t in texts]

        return texts

    total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"  Total eval batch size = {total_batch_size}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(dataset) // total_batch_size + 1), disable=not accelerator.is_local_main_process)

    model.eval()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": args.val_max_target_length,
        "min_length": args.val_min_target_length,
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_beams,
        "return_dict_in_generate": True,
        "output_scores": True
    }

    f = open(args.result_dir + "/output.txt", "w")
    results = []

    for batch in eval_dataloader:
        progress_bar.update(1)
        with torch.no_grad():

            inputs = batch['input_ids']
            inputs = accelerator.pad_across_processes(batch["input_ids"], dim=1, pad_index=tokenizer.pad_token_id)
            inputs = accelerator.gather(inputs).cpu().numpy()
            decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
            decoded_inputs = postprocess_text(decoded_inputs)

            labels = batch["labels"]
            labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
            labels = accelerator.gather(labels).cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_labels = postprocess_text(decoded_labels)

            outputs = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens, scores = outputs['sequences'], outputs['sequences_scores']

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_preds = postprocess_text(decoded_preds)

            scores = accelerator.gather(scores).cpu().tolist()

            ex_num = len(decoded_inputs)
            for idx in range(ex_num):
                dialogue = decoded_inputs[idx].replace('\r\n', '\n').split('\n')
                dialogue = '\n\t'.join([str(i+1) + " : " + sent for i, sent in enumerate(dialogue)])
                summary = decoded_labels[idx].replace('\n', " ")
                f.write("dialogue: \n\t%s\nsummary: \n\t%s\n\n" % (dialogue, summary))
                output_result = []
                for i in range(args.num_beams):
                    f.write("beam %d:\t%.4f\t%s\n" % (i+1,
                            scores[args.num_beams*idx+i],
                            decoded_preds[args.num_beams*idx+i].replace("\n", " ")))
                    output_result.append(decoded_preds[args.num_beams*idx+i])
                f.write('\n\n')
                results.append(output_result)
    f.close()
    return results
