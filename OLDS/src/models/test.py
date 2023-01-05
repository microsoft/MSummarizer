import os
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from others.metric import Metric_CLS
from others.dataset import get_omission_datasets
from models.detection import DetectionModel
from models.predict import build_predicted_datasets
from nltk import word_tokenize
from torch.utils.data import DataLoader
from transformers.utils import WEIGHTS_NAME
from transformers import (
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
)


def test(args, logger, accelerator, tokenizer=None, model=None, dataset=None, wordR_metric_split="test"):

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)

    if model is None:
        accelerator.free_memory()
        # model = AutoModelForSequenceClassification.from_pretrained(args.model)
        model = DetectionModel(args, os.path.join(args.checkpoint, WEIGHTS_NAME))

        # Prepare model with our `accelerator`.
        model = accelerator.prepare(model)

    if dataset is None:
        dataset = get_omission_datasets(args, accelerator, tokenizer)['test']

    if args.mode == 'pair':
        data_collator = DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )
    elif args.mode == 'seq':
        data_collator = DataCollatorForTokenClassification(
            tokenizer,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )

    eval_dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Prepare everything with our `accelerator`.
    eval_dataloader = accelerator.prepare(eval_dataloader)

    # Metric
    metric = Metric_CLS()
    if args.num_labels > 2:
        metric_sep = Metric_CLS()

    total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"  Total eval batch size = {total_batch_size}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(dataset) // total_batch_size + 1), disable=not accelerator.is_local_main_process)

    model.eval()
    samples_seen = 0

    predicted_labels = []
    raw_labels = []
    
    for step, batch in enumerate(eval_dataloader):
        progress_bar.update(1)
        with torch.no_grad():
            outputs = model(**batch, mode=args.mode, cls_token_id=tokenizer.cls_token_id)

        predictions = outputs.logits.argmax(-1) if args.num_labels > 1 else outputs.logits.squeeze()

        if args.mode != "pair":
            pred_list = torch.split(predictions, outputs.labels.ne(-100).sum(-1).tolist())
            predictions = pad_sequence(pred_list, batch_first=True, padding_value=-100)
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            
            labels = accelerator.pad_across_processes(outputs.labels, dim=1, pad_index=-100)
        else:
            labels = outputs.labels

        predictions, references = accelerator.gather((predictions, labels))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]

        refs = references[references.ne(-100)]
        preds = predictions[predictions.ne(-100)]
        
        if args.num_labels > 2:
            metric_sep.add_batch(preds, refs)

        metric.add_batch(preds.eq(2).long(), refs.eq(2).long())

        predicted_labels.extend(preds.tolist())
        raw_labels.extend(refs.tolist())

    if args.num_labels > 2:
        result = metric_sep.compute(mode=None, labels=list(range(args.num_labels)))
        logger.info(result)

    result = metric.compute()

    if accelerator.is_local_main_process:
        predicted_data = build_predicted_datasets(args, accelerator, tokenizer,
            predicted_labels[::-1], raw_labels[::-1], wordR_metric_split)
        total_wordR = 0.
        total_num = 0
        for ex in predicted_data:
            turns = ex['dialogue'].replace('\r\n', '\n').split('\n')
            pred_labels = ex['omission_labels']
            omit_words = ex['omission_words']
            for i in range(len(pred_labels)):
                if args.domain != 'all' and args.domain != ex['preds'][i]['source']:
                    continue
                pred_words = set(word_tokenize(" ".join([turns[idx] for idx in pred_labels[i]]).lower()))
                gold_words = set(sum(omit_words[i], []))
                cnt = 0
                for w in gold_words:
                    if w in pred_words:
                        cnt += 1
                wordR = cnt / len(gold_words) if len(gold_words) > 0 else 1.
                total_wordR += wordR
                total_num += 1

        result['WordR'] = round(total_wordR / total_num, 6)
    logger.info(result)

    return result


def test_all(args, logger, accelerator):

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)

    accelerator.free_memory()
    # model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model = DetectionModel(args, os.path.join(args.checkpoint, WEIGHTS_NAME))

    # Prepare model with our `accelerator`.
    model = accelerator.prepare(model)

    dataset = get_omission_datasets(args, accelerator, tokenizer)

    # test(args, logger, accelerator, tokenizer, model, dataset['validation'], "validation")
    test(args, logger, accelerator, tokenizer, model, dataset['test'], "test")
