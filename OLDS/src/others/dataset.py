from datasets import load_dataset, load_from_disk, DatasetDict
from nltk import sent_tokenize
from random import shuffle
from others.metric import Metric


def get_raw_datasets(args):

    summarization_name_mapping = {
        "cnndm": ("article", "highlights"),
        "samsum": ("dialogue", "summary")
    }

    # Get the datasets. Downloading and loading a dataset from the hub.
    if args.dataset == 'cnndm':
        raw_datasets = load_dataset('ccdv/cnn_dailymail', '3.0.0')
    elif args.dataset == 'samsum':
        raw_datasets = load_dataset('samsum')
    else:
        raw_datasets = load_from_disk(args.dataset)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(args.dataset, None)
    text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]

    return raw_datasets, text_column, summary_column


def get_datasets(args, accelerator, tokenizer):

    raw_datasets, text_column, summary_column = get_raw_datasets(args)

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]

        if args.prefix != "":
            inputs = [args.prefix + " " + doc for doc in inputs]

        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=False, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=False, truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

    return processed_datasets


def get_omission_datasets(args, accelerator, tokenizer):

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

    def pair_func(examples):

        dialogues = examples['dialogue']
        preds = examples['preds']
        omission_labels = examples['omission_labels']
        oracle_labels = examples['oracle_labels']

        dial_inputs = []
        pred_inputs = []
        labels = []

        for dialogue, pred, label, gold in zip(dialogues, preds, omission_labels, oracle_labels):
            sents = dialogue.replace('\r\n', '\n').split('\n')
            for i, sent in enumerate(sents):
                for j, p in enumerate(pred):
                    if args.domain != 'all' and args.domain != p['source']:
                        continue
                    if i in gold:
                        if i in label[j]:
                            la = 2
                        else:
                            la = 1
                    else:
                        la = 0
                    dial_inputs.append(sent)
                    pred_inputs.append(p['pred'])
                    labels.append(la)
        
        if len(dial_inputs) > 0:
            model_inputs = tokenizer(dial_inputs, pred_inputs,
                                     max_length=args.max_source_length,
                                     padding=False, truncation=True)

            model_inputs["labels"] = labels
        else:
            model_inputs = {}

        return model_inputs

    def seq_func(examples):

        dialogues = examples['dialogue']
        preds = examples['preds']
        omission_labels = examples['omission_labels']
        oracle_labels = examples['oracle_labels']

        dial_inputs = []
        pred_inputs = []
        labels = []

        for dialogue, pred, label, gold in zip(dialogues, preds, omission_labels, oracle_labels):
            new_dial, sent_num = _add_start_end_token(dialogue)
            for i, p in enumerate(pred):
                if args.domain != 'all' and args.domain != p['source']:
                    continue
                dial_inputs.append(new_dial)
                pred_inputs.append(p['pred'])
                label_flat = []
                for j in range(sent_num):
                    if j in gold:
                        if j in label[i]:
                            label_flat.append(2)
                        else:
                            label_flat.append(1)
                    else:
                        label_flat.append(0)
                labels.append(label_flat)

        if len(pred_inputs) > 0:
            model_inputs = tokenizer(pred_inputs, dial_inputs,
                                     max_length=args.max_source_length,
                                     padding=False, truncation=True)

            model_inputs["labels"] = list(
                map(lambda x: _truncate_cls_labels(x),
                    zip(labels, model_inputs['input_ids']))
            )
        else:
            model_inputs = {}

        return model_inputs

    def span_func(examples):
        dialogues = examples['dialogue']
        preds = examples['preds']
        omission_labels = examples['omission_labels']
        oracle_labels = examples['oracle_labels']

        dial_inputs = []
        pred_inputs = []
        labels = []

        for dialogue, pred, label, gold in zip(dialogues, preds, omission_labels, oracle_labels):
            new_dial, _ = _add_start_end_token(dialogue)
            for i, p in enumerate(pred):
                if args.domain != 'all' and args.domain != p['source']:
                    continue
                dial_inputs.append(new_dial)
                pred_inputs.append(p['pred'])
                la = [e for e in gold if e not in label[i]]
                labels.append(label[i] + [-1] + la)

        if len(pred_inputs) > 0:
            model_inputs = tokenizer(pred_inputs, dial_inputs,
                                     max_length=args.max_source_length,
                                     padding=False, truncation=True)

            model_inputs["labels"] = list(
                map(lambda x: _truncate_cls_indexs(x),
                    zip(labels, model_inputs['input_ids']))
            )
        else:
            model_inputs = {}
        return model_inputs

    if args.mode == 'pair':
        preprocess_function = pair_func
    elif args.mode == 'seq':
        preprocess_function = seq_func
    else:
        preprocess_function = span_func

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

    return processed_datasets


def get_post_edit_training_datasets(args, accelerator, tokenizer, logger=None,
                                    split_num=2, cut_ratio=0., add_ratio=0., keep_size=False):

    raw_datasets = load_from_disk(args.dataset)
    sep_token = tokenizer.sep_token
    eos_token = tokenizer.eos_token
    sep_txt = eos_token + sep_token if sep_token is not None else eos_token

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]

        return preds, labels

    def preprocess_function_eval(examples):

        dialogues = examples['dialogue']
        summaries = examples['summary']
        preds = examples['preds']
        omission_labels = examples['omission_labels']
        if 'oracle_labels' in examples:
            oracle_labels = examples['oracle_labels']
        else:
            oracle_labels = examples['new_oracle_labels']

        omit_inputs = []
        pred_inputs = []
        targets = []

        for dialogue, summ, pred, label, gold in zip(dialogues, summaries, preds, omission_labels, oracle_labels):
            sents = dialogue.replace('\r\n', '\n').split('\n')
            
            rest_sent = [[j for j in range(len(sents)) if j not in label[i]] for i in range(len(pred))]

            for x in label: shuffle(x)
            for x in rest_sent: shuffle(x)

            cutted_label = [sorted(x[:round(len(x) * (1.-cut_ratio))]) for x in label]
            if keep_size:
                label = [sorted(cutted_label[i] + rest_sent[i][:round(len(x) * add_ratio)]) for i, x in enumerate(label)]
            else:
                label = [sorted(cutted_label[i] + rest_sent[i][:round(len(rest_sent[i]) * add_ratio)]) for i, x in enumerate(label)]

            for i, p in enumerate(pred):
                if args.domain != 'all' and args.domain != p['source']:
                    continue
                if split_num == 1:
                    omit_inputs.append(dialogue)
                elif split_num == 2:
                    omit_sent = "\n".join([sents[j] for j in label[i]])
                    other_sent = "\n".join([sents[j] for j in range(len(sents)) if j not in label[i]])
                    omit_inputs.append(omit_sent + sep_txt + other_sent)
                else:
                    gd = gold if 'oracle_labels' in examples else gold[i]
                    omit_sent = "\n".join([sents[j] for j in label[i]])
                    oracle_sent = "\n".join([sents[j] for j in range(len(sents)) if j in gd and j not in label[i]])
                    other_sent = "\n".join([sents[j] for j in range(len(sents)) if j not in gd and j not in label[i]])
                    omit_inputs.append(omit_sent + sep_txt + oracle_sent + sep_txt + other_sent)
                pred_inputs.append(p['pred'])
                targets.append(summ)

        decoded_preds, decoded_labels = postprocess_text(pred_inputs, targets)

        metric.add_batch(preds=decoded_preds, refs=decoded_labels)

        if args.prefix != "":
            pred_inputs = [args.prefix + " " + doc for doc in pred_inputs]

        model_inputs = tokenizer(pred_inputs, omit_inputs,
                                 max_length=args.max_source_length,
                                 padding=False, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=False, truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def preprocess_function_train(examples):

        dialogues = examples['dialogue']
        summaries = examples['summary']
        preds = examples['preds']
        omission_labels = examples['omission_labels']
        oracle_labels = examples['oracle_labels']

        omit_inputs = []
        pred_inputs = []
        targets = []

        for dialogue, summ, pred, label, gold in zip(dialogues, summaries, preds, omission_labels, oracle_labels):
            sents = dialogue.replace('\r\n', '\n').split('\n')
            for i, p in enumerate(pred):
                if split_num == 1:
                    omit_inputs.append(dialogue)
                elif split_num == 2:
                    omit_sent = "\n".join([sents[j] for j in label[i]])
                    other_sent = "\n".join([sents[j] for j in range(len(sents)) if j not in label[i]])
                    omit_inputs.append(omit_sent + sep_txt + other_sent)
                else:
                    omit_sent = "\n".join([sents[j] for j in label[i]])
                    oracle_sent = "\n".join([sents[j] for j in range(len(sents)) if j in gold and j not in label[i]])
                    other_sent = "\n".join([sents[j] for j in range(len(sents)) if j not in gold and j not in label[i]])
                    omit_inputs.append(omit_sent + sep_txt + oracle_sent + sep_txt + other_sent)
                pred_inputs.append(p['pred'])
                targets.append(summ)

        if args.prefix != "":
            pred_inputs = [args.prefix + " " + doc for doc in pred_inputs]

        model_inputs = tokenizer(pred_inputs, omit_inputs,
                                 max_length=args.max_source_length,
                                 padding=False, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=False, truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    if 'train' in raw_datasets:
        with accelerator.main_process_first():
            train_set = raw_datasets['train'].map(
                preprocess_function_train,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on training dataset",
            )

    metric = Metric()

    with accelerator.main_process_first():
        dev_set = raw_datasets['validation'].map(
            preprocess_function_eval,
            batched=True,
            remove_columns=raw_datasets["validation"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on validation dataset",
        )

    # Calculate metrics
    if logger is not None:
        result = metric.compute()
        logger.info("Validation set raw results:")
        logger.info(result)

    metric = Metric()

    with accelerator.main_process_first():
        test_set = raw_datasets['test'].map(
            preprocess_function_eval,
            batched=True,
            remove_columns=raw_datasets["test"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on testing dataset",
        )

    # Calculate metrics
    if logger is not None:
        result = metric.compute()
        logger.info("Test set raw results:")
        logger.info(result)

    new_set = DatasetDict({
        "train": train_set if 'train' in raw_datasets else None,
        "validation": dev_set,
        "test": test_set
    })
    return new_set