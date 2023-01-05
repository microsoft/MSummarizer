import os
from time import sleep
from datasets import Dataset, DatasetDict, load_from_disk
from seq2seq.predict import predict
from seq2seq.train import train
from nltk.tokenize import sent_tokenize
from others.metric import Metric
from others.dataset import get_datasets, get_raw_datasets
from transformers import AutoTokenizer


def greedy_selection(doc_sents, summ, metric):

    summ = summ.lower()
    doc_sents = [s.lower() for s in doc_sents]

    refs = [summ for _ in range(len(doc_sents))]
    sent_rouge = metric.compute_rouge(doc_sents, refs, record=False, use_recall=True)
    scores = sorted([(i, sum(ex.values())) for i, ex in enumerate(sent_rouge)],
                    key=lambda x: x[1], reverse=True)

    max_score = [0. for _ in range(len(sent_rouge[0]))]
    selected = []
    for i, _ in scores:
        buff = selected + [i]
        for j in range(len(buff)-1, 0, -1):
            if buff[j-1] > buff[j]:
                buff[j-1], buff[j] = buff[j], buff[j-1]
            else:
                break
        cand = " ".join(doc_sents[j] for j in buff)
        score = list(metric.compute_rouge([cand], [summ], record=False, use_recall=True)[0].values())
        for j in range(len(max_score)):
            if score[j] > max_score[j]:
                max_score = score
                selected = buff
                break

    return selected


def build_omission_data(args, logger, accelerator, tokenizer=None):

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    os.makedirs(args.save_path, exist_ok=True)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    dataset = get_datasets(args, accelerator, tokenizer)
    raw_datasets, text_column, summary_column = get_raw_datasets(args)

    # split training data
    split_size = len(dataset['train']) // args.data_split_size
    output_dir = args.output_dir
    model_path = args.model
    total_preds_train = []
    for i in range(args.data_split_size):
        if i < args.data_split_size-1:
            train_dataset = dataset['train'].select(
                list(range(i*split_size)) + list(range((i+1)*split_size, len(dataset['train'])))
            )
            eval_dataset = dataset['train'].select(range(i*split_size, (i+1)*split_size))
        else:
            train_dataset = dataset['train'].select(list(range(i*split_size)))
            eval_dataset = dataset['train'].select(range(i*split_size, len(dataset['train'])))

        new_dataset = {
            "train": train_dataset,
            "validation": eval_dataset,
            "test": eval_dataset
        }

        # training
        args.output_dir = output_dir + "/" + str(i+1)
        if not os.path.exists(args.output_dir + "/round_final"):
            logger.info("Round %d\nStart Training!" % (i+1))
            args.model = model_path
            train(args, logger, accelerator, tokenizer=tokenizer, dataset=new_dataset, save_final=True)
            # wait model to be saved
            sleep(10)

        # predicting
        args.model = args.output_dir + "/round_final"

        logger.info("\nStart Predicting!")
        preds = predict(args, logger, accelerator, tokenizer=tokenizer, dataset=new_dataset['test'])
        total_preds_train.extend(preds[:len(eval_dataset)])
        logger.info("\n\n\n")

    # eval test data
    args.output_dir = output_dir + "/all"

    # training
    if not os.path.exists(args.output_dir + "/round_final"):
        logger.info("Round All\nStart Training!")
        args.model = model_path
        train(args, logger, accelerator, tokenizer=tokenizer, dataset=dataset, save_final=True)
        # wait model to be saved
        sleep(10)

    # predicting
    args.model = args.output_dir + "/round_final"

    logger.info("\nStart Predicting!")
    total_preds_eval = predict(args, logger, accelerator, tokenizer=tokenizer, dataset=dataset['validation'])
    total_preds_test = predict(args, logger, accelerator, tokenizer=tokenizer, dataset=dataset['test'])
    logger.info("\n\n\n")

    train_set = {
        text_column: [ex[text_column] for ex in raw_datasets['train']],
        summary_column: [ex[summary_column] for ex in raw_datasets['train']],
        'id': [ex['id'] for ex in raw_datasets['train']],
        'preds': total_preds_train
    }
    eval_set = {
        text_column: [ex[text_column] for ex in raw_datasets['validation']],
        summary_column: [ex[summary_column] for ex in raw_datasets['validation']],
        'id': [ex['id'] for ex in raw_datasets['validation']],
        'preds': total_preds_eval[:len(dataset['validation'])]
    }
    test_set = {
        text_column: [ex[text_column] for ex in raw_datasets['test']],
        summary_column: [ex[summary_column] for ex in raw_datasets['test']],
        'id': [ex['id'] for ex in raw_datasets['test']],
        'preds': total_preds_test[:len(dataset['test'])]
    }

    new_set = DatasetDict({
        "train": Dataset.from_dict(train_set),
        "validation": Dataset.from_dict(eval_set),
        "test": Dataset.from_dict(test_set)
    })

    strategy_token = "sample" if args.do_sample else "beam"
    new_set.save_to_disk(args.save_path + '/omission_' + strategy_token + '_preds.save')
    accelerator.free_memory()


# Omission labels construction
def build_omission_label(args):
    stop_word_set = set()
    with open('src/others/stop_word_list', 'r') as f:
        for word in f:
            stop_word_set.add(word.strip())

    _, text_column, summary_column = get_raw_datasets(args)
    dataset = load_from_disk(args.save_path + '/omission_preds.save')
    metric = Metric()

    def word_tokenize(s):
        s = "".join([ch if (ch>='a' and ch<='z') or (ch>='0' and ch<='9') or ch == '\'' else ' '
                    for ch in s.lower().replace('\n', ' ')]).split()
        res = set()
        for word in s:
            if word not in stop_word_set:
                res.add(word)
        return res

    def process_function(ex):

        doc = ex[text_column]
        summ = ex[summary_column]
        preds = ex['preds']

        if text_column == 'dialogue':
            doc_sents = doc.replace('\r\n', '\n').split('\n')
        else:
            doc_sents = sent_tokenize(doc)

        summ_tokens = word_tokenize(summ)
        overlap_tokens = [word_tokenize(s) & summ_tokens for s in doc_sents]

        gold_labels = greedy_selection(doc_sents, summ, metric)
        pred_labels = []
        omissions = []
        omission_words = []
        for pred in preds:
            pred_tokens = word_tokenize(pred['pred'])
            pred_label = greedy_selection(doc_sents, pred['pred'], metric)
            pred_labels.append(pred_label)
            omission_word = dict()
            for la in gold_labels:
                for word in overlap_tokens[la]:
                    if word not in pred_tokens:
                        if la in omission_word:
                            omission_word[la].append(word)
                        else:
                            omission_word[la] = [word]

            # remove redundancy
            omission_word = sorted(list(omission_word.items()), key=lambda x: len(x[1]), reverse=True)
            omission, omission_w = [], []
            word_set = set()
            for la, w in omission_word:
                ores = False
                for word in w:
                    if word not in word_set:
                        ores = True
                        word_set.add(word)
                if ores:
                    omission.append(la)
                    omission_w.append(w)

            reorder = sorted(zip(omission, omission_w), key=lambda x: x[0])
            omission_words.append([x[1] for x in reorder])
            omissions.append([x[0] for x in reorder])

        return {
            'oracle_labels': gold_labels,
            'pred_labels': pred_labels,
            'omission_labels': omissions,
            'omission_words': omission_words
        }

    processed_datasets = dataset.map(
        process_function,
        batched=False,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=False,
        desc="Running processing on dataset",
    )

    processed_datasets.save_to_disk(args.save_path + '/omission.save')


def group_omission_data(args, models=['bart_large', 'bart_base', 't5_base', 't5_small', 'baseline', 'pegasus']):

    raw_datasets, text_column, summary_column = get_raw_datasets(args)

    total_preds_train = [[] for _ in range(len(raw_datasets['train']))]
    total_preds_eval = [[] for _ in range(len(raw_datasets['validation']))]
    total_preds_test = [[] for _ in range(len(raw_datasets['test']))]
    for model in models:
        for strategy_token in ['beam', 'sample']:
            p = args.save_path + '/' + model + '/omission_' + strategy_token + '_preds.save'
            data = load_from_disk(p)
            if model != 'pegasus':
                for i, preds in enumerate(data['train']['preds']):
                    total_preds_train[i].extend([{'source': model, 'strategy': strategy_token, 'pred': pred} for pred in preds])
            for i, preds in enumerate(data['validation']['preds']):
                total_preds_eval[i].append({'source': model, 'strategy': strategy_token, 'pred': preds[0]})
            for i, preds in enumerate(data['test']['preds']):
                total_preds_test[i].append({'source': model, 'strategy': strategy_token, 'pred': preds[0]})

    train_set = {
        text_column: [ex[text_column] for ex in raw_datasets['train']],
        summary_column: [ex[summary_column] for ex in raw_datasets['train']],
        'id': [ex['id'] for ex in raw_datasets['train']],
        'preds': total_preds_train
    }
    eval_set = {
        text_column: [ex[text_column] for ex in raw_datasets['validation']],
        summary_column: [ex[summary_column] for ex in raw_datasets['validation']],
        'id': [ex['id'] for ex in raw_datasets['validation']],
        'preds': total_preds_eval
    }
    test_set = {
        text_column: [ex[text_column] for ex in raw_datasets['test']],
        summary_column: [ex[summary_column] for ex in raw_datasets['test']],
        'id': [ex['id'] for ex in raw_datasets['test']],
        'preds': total_preds_test
    }

    raw_train_set = Dataset.from_dict(train_set)

    def edit_dis(x, y):
        f = [[0 for _ in range(len(y)+1)] for _ in range(len(x)+1)]
        for i in range(1, len(x)+1):
            f[i][0] = i
        for i in range(1, len(y)+1):
            f[0][i] = i
        for i in range(1, len(x)+1):
            for j in range(1, len(y)+1):
                f[i][j] = min(f[i-1][j], f[i][j-1])+1
                if x[i-1] == y[j-1]:
                    f[i][j] = min(f[i][j], f[i-1][j-1])
                else:
                    f[i][j] = min(f[i][j], f[i-1][j-1]+1)
        return f[len(x)][len(y)]

    def process(ex):
        preds_list = ex['preds']
        dis = [[0 for _ in range(len(preds_list))] for _ in range(len(preds_list))]
        for i in range(len(preds_list)):
            for j in range(i+1, len(preds_list)):
                if preds_list[i]['pred'] != preds_list[j]['pred']:
                    dis[i][j] = dis[j][i] = edit_dis(preds_list[i]['pred'], preds_list[j]['pred'])
        dis = [[i for i in t if i > 0] for t in dis]
        mean_dis = [sum(t) / len(t) for t in dis]
        sorted_preds = sorted(zip(mean_dis, preds_list), reverse=True, key=lambda x: x[0])
        new_preds = dict()
        for _, item in sorted_preds:
            if len(new_preds) >= 10:
                break
            if item['pred'] in new_preds:
                continue
            new_preds[item['pred']] = item

        return {
            'preds': list(new_preds.values())
        }

    # remove duplicate preds
    processed_train_set = raw_train_set.map(
        process,
        batched=False,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=False,
        remove_columns=['preds'],
        desc="Running processing on dataset",
    )

    processed_datasets = DatasetDict({
        "train": processed_train_set,
        "validation": Dataset.from_dict(eval_set),
        "test": Dataset.from_dict(test_set)
    })
    processed_datasets.save_to_disk(args.save_path + '/omission_preds.save')
