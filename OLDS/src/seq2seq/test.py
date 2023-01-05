import torch
import nltk
import math
import statistics
import numpy as np
from tqdm.auto import tqdm
from others.metric import Metric
from others.dataset import get_datasets
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


def test(args, logger, accelerator, tokenizer=None, model=None, dataset=None):

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

    # Metric
    metric = Metric()

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

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
        "num_beams": args.num_beams,
        "do_sample": args.do_sample
    }

    ref_loss_results = []
    if args.output_attentions:
        attn_output = {
            "attentions": [],
            "input_token_ids": [],
            "attention_masks": []
        }

    for batch in eval_dataloader:
        progress_bar.update(1)
        with torch.no_grad():

            # Get reference related losses.
            outputs = accelerator.unwrap_model(model)(
                output_attentions=args.output_attentions,
                return_dict=True,
                **batch
            )
            ref_loss_results.append(accelerator.gather(outputs['loss']).mean().cpu().item())

            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            if args.output_attentions:
                cross_attention = torch.cat([t.mean((1, 2)).unsqueeze_(1) for t in outputs["cross_attentions"]], 1)
                cross_attention = accelerator.pad_across_processes(cross_attention, dim=2, pad_index=0.)
                cross_attention = accelerator.gather(cross_attention)

                input_ids = accelerator.pad_across_processes(batch["input_ids"], dim=1, pad_index=tokenizer.pad_token_id)
                input_ids = accelerator.gather(input_ids).cpu()
                attention_mask = accelerator.pad_across_processes(batch["attention_mask"], dim=1, pad_index=0)
                attention_mask = accelerator.gather(attention_mask).cpu()

                attn_output['attentions'].extend([item for item in cross_attention])
                attn_output["input_token_ids"].extend([item for item in input_ids])
                attn_output["attention_masks"].extend([item for item in attention_mask])

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]
            labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            metric.add_batch(preds=decoded_preds, refs=decoded_labels)

    # Saving attention scores
    if args.output_attentions:
        with open(args.output_attention_path + "/attn_output.pt", "wb") as f:
            torch.save(attn_output, f)

    # Calculate metrics
    result = metric.compute()
    ref_ppl = math.exp(statistics.mean(ref_loss_results))
    result['ppl']['ref_perplexity'] = round(ref_ppl, 4)
    logger.info(result)
    return result
