import os
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from others.metric import Metric_CLS
from others.dataset import get_omission_datasets
from models.detection import DetectionModel
from torch.utils.data import DataLoader
from datasets import load_from_disk, DatasetDict
from transformers.utils import WEIGHTS_NAME
from sklearn.metrics import confusion_matrix
from transformers import (
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
)


def build_predicted_datasets(args, accelerator, tokenizer, predictions, references, split):

    raw_datasets = load_from_disk(args.dataset)
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token

    def _add_start_end_token(x):
        x_list = x.replace('\r\n', '\n').split('\n')
        sent_num = len(x_list)
        return cls_token + (sep_token + cls_token).join(x_list), sent_num

    def _truncate_cls_labels(x):
        label, input_id = x
        num_cls = 0
        for i, item in enumerate(input_id):
            if i > 0 and item == tokenizer.cls_token_id:
                num_cls += 1
        return label[:num_cls]

    def _truncate_cls_indexs(x):
        label, input_id = x
        num_cls = 0
        for i, item in enumerate(input_id):
            if i > 0 and item == tokenizer.cls_token_id:
                num_cls += 1
        return [idx for idx in label if idx < num_cls]

    def pair_func(ex):

        dialogue = ex['dialogue']
        summary = ex['summary']
        preds = ex['preds']
        labels = ex['omission_labels']
        oracles = ex['oracle_labels']
        pred_labels = ex['pred_labels']
        omission_words = ex['omission_words']

        sents = dialogue.replace('\r\n', '\n').split('\n')
        new_omission_labels = [[] for _ in range(len(preds))]
        new_oracle_labels = [[] for _ in range(len(preds))]

        for i in range(len(sents)):
            for j, p in enumerate(preds):
                if args.domain != 'all' and args.domain != p['source']:
                    continue
                if i in oracles:
                    if i in labels[j]:
                        la = 2
                    else:
                        la = 1
                else:
                    la = 0
                assert la == references[-1]
                if predictions[-1] == 2:
                    new_omission_labels[j].append(i)
                    new_oracle_labels[j].append(i)
                elif predictions[-1] == 1:
                    new_oracle_labels[j].append(i)
                references.pop()
                predictions.pop()

        return {
            "dialogue": dialogue,
            "summary": summary,
            "preds": preds,
            "new_oracle_labels": new_oracle_labels,
            "pred_labels": pred_labels,
            "omission_labels": new_omission_labels,
            "omission_words": omission_words
        }

    def seq_func(ex):

        dialogue = ex['dialogue']
        summary = ex['summary']
        preds = ex['preds']
        labels = ex['omission_labels']
        oracles = ex['oracle_labels']
        pred_labels = ex['pred_labels']
        omission_words = ex['omission_words']

        dial_inputs = []
        pred_inputs = []
        labels_flat = []

        new_dial, sent_num = _add_start_end_token(dialogue)
        for i, p in enumerate(preds):
            if args.domain != 'all' and args.domain != p['source']:
                continue
            dial_inputs.append(new_dial)
            pred_inputs.append(p['pred'])
            label_flat = []
            for j in range(sent_num):
                if j in oracles:
                    if j in labels[i]:
                        label_flat.append(2)
                    else:
                        label_flat.append(1)
                else:
                    label_flat.append(0)
            labels_flat.append(label_flat)

        model_inputs = tokenizer(pred_inputs, dial_inputs,
                                 max_length=args.max_source_length,
                                 padding=False, truncation=True)

        model_inputs["labels"] = list(
            map(lambda x: _truncate_cls_labels(x),
                zip(labels_flat, model_inputs['input_ids']))
        )

        new_omission_labels = [[] for _ in range(len(preds))]
        new_oracle_labels = [[] for _ in range(len(preds))]

        label_i = 0
        for i, p in enumerate(preds):
            if args.domain != 'all' and args.domain != p['source']:
                continue
            for j, la in enumerate(model_inputs["labels"][label_i]):
                assert la == references[-1]
                if predictions[-1] == 2:
                    new_omission_labels[i].append(j)
                    new_oracle_labels[i].append(j)
                elif predictions[-1] == 1:
                    new_oracle_labels[i].append(j)
                references.pop()
                predictions.pop()
            label_i += 1

        return {
            "dialogue": dialogue,
            "summary": summary,
            "preds": preds,
            "new_oracle_labels": new_oracle_labels,
            "pred_labels": pred_labels,
            "omission_labels": new_omission_labels,
            "omission_words": omission_words
        }

    def span_func(ex):
        dialogue = ex['dialogue']
        summary = ex['summary']
        preds = ex['preds']
        labels = ex['omission_labels']
        oracles = ex['oracle_labels']
        pred_labels = ex['pred_labels']
        omission_words = ex["omission_words"]

        dial_inputs = []
        pred_inputs = []
        new_labels = []

        new_dial, sent_num = _add_start_end_token(dialogue)
        for i, p in enumerate(preds):
            if args.domain != 'all' and args.domain != p['source']:
                continue
            dial_inputs.append(new_dial)
            pred_inputs.append(p['pred'])
            la = [e for e in oracles if e not in labels[i]]
            new_labels.append(labels[i] + [-1] + la)

        model_inputs = tokenizer(pred_inputs, dial_inputs,
                                 max_length=args.max_source_length,
                                 padding=False, truncation=True)

        model_inputs["labels"] = list(
            map(lambda x: _truncate_cls_indexs(x),
                zip(new_labels, model_inputs['input_ids']))
        )

        labels_flat = []
        for i in range(len(model_inputs['labels'])):
            label_flat = []
            index_1 = model_inputs['labels'][i].index(-1)
            for j in range(sent_num):
                if j in oracles:
                    if j in model_inputs['labels'][i][:index_1]:
                        label_flat.append(2)
                    elif j in model_inputs['labels'][i][index_1:]:
                        label_flat.append(1)
                    else:
                        label_flat.append(0)
                else:
                    label_flat.append(0)
            labels_flat.append(label_flat)
        
        model_inputs["labels"] = list(
            map(lambda x: _truncate_cls_labels(x),
                zip(labels_flat, model_inputs['input_ids']))
        )

        new_omission_labels = [[] for _ in range(len(preds))]
        new_oracle_labels = [[] for _ in range(len(preds))]

        label_i = 0
        for i, p in enumerate(preds):
            if args.domain != 'all' and args.domain != p['source']:
                continue
            for j, la in enumerate(model_inputs["labels"][label_i]):
                assert la == references[-1]
                if predictions[-1] == 2:
                    new_omission_labels[i].append(j)
                    new_oracle_labels[i].append(j)
                elif predictions[-1] == 1:
                    new_oracle_labels[i].append(j)
                references.pop()
                predictions.pop()
            label_i += 1

        return {
            "dialogue": dialogue,
            "summary": summary,
            "preds": preds,
            "new_oracle_labels": new_oracle_labels,
            "pred_labels": pred_labels,
            "omission_labels": new_omission_labels,
            "omission_words": omission_words
        }

    if args.mode == 'pair':
        preprocess_function = pair_func
    elif args.mode == 'seq':
        preprocess_function = seq_func
    else:
        preprocess_function = span_func

    with accelerator.main_process_first():
        processed_datasets = raw_datasets[split].map(
            preprocess_function,
            batched=False,
            remove_columns=raw_datasets[split].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

    assert len(references) == 0
    return processed_datasets


def predict(args, logger, accelerator, tokenizer=None, model=None, dataset=None):

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

    predicted_labels = []
    raw_labels = []
    model.eval()
    samples_seen = 0
    for step, batch in enumerate(eval_dataloader):
        progress_bar.update(1)
        with torch.no_grad():
            outputs = model(**batch, mode=args.mode, cls_token_id=tokenizer.cls_token_id)

        assert outputs.logits.size(-1) >= 2
        logits = F.softmax(outputs.logits, -1) if args.mode != 'span' else outputs.logits.float()

        if args.confidence_ratio is not None:
            predictions = logits[:, :-1].argmax(-1)
            predictions[logits[:, -1] > args.confidence_ratio] = outputs.logits.size(-1) - 1
        else:
            predictions = logits.argmax(-1)
        
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

    result = metric.compute()
    logger.info(result)
    if args.num_labels > 2:
        result = metric_sep.compute(mode=None, labels=list(range(args.num_labels)))
        logger.info(result)

    logger.info(confusion_matrix(raw_labels, predicted_labels, labels=list(range(args.num_labels))))
    return predicted_labels, raw_labels


def predict_all(args, logger, accelerator):

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)

    accelerator.free_memory()
    # model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model = DetectionModel(args, os.path.join(args.checkpoint, WEIGHTS_NAME))

    # Prepare model with our `accelerator`.
    model = accelerator.prepare(model)

    dataset = get_omission_datasets(args, accelerator, tokenizer)

    # predictions

    preds, refs = predict(args, logger, accelerator, tokenizer, model, dataset['validation'])

    if accelerator.is_local_main_process:
        val_data = build_predicted_datasets(args, accelerator, tokenizer, preds[::-1], refs[::-1], "validation")

    preds, refs = predict(args, logger, accelerator, tokenizer, model, dataset['test'])

    if accelerator.is_local_main_process:
        test_data = build_predicted_datasets(args, accelerator, tokenizer, preds[::-1], refs[::-1], "test")

        new_data = DatasetDict({
            "validation": val_data,
            "test": test_data
        })
        new_data_path = '/'.join(args.dataset.split('/')[:-1])
        new_data.save_to_disk(new_data_path + '/omission_new.save')
