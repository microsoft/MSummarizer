from seq2seq.test import test
from seq2seq.train import train, resize_position_embeddings
from time import sleep
from others.dataset import get_post_edit_training_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration
)


def post_edit_train(args, logger, accelerator):

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    if isinstance(model, BartForConditionalGeneration) and \
        args.max_source_length > model.config.max_position_embeddings:
        resize_position_embeddings(model.model, args.max_source_length)
        model.config.max_position_embeddings = args.max_source_length

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Prepare model with our `accelerator`.
    model = accelerator.prepare(model)

    # Load datasets.
    dataset = get_post_edit_training_datasets(args, accelerator, tokenizer,
        split_num=args.post_edit_split_num)

    # training
    results = train(args, logger, accelerator, tokenizer, model, dataset)

    # evaluation
    sleep(10)
    best_step = -1
    best_result = (-1, -1)
    for step, result in results.items():
        rouge_score = result['rouge']['rouge1'] + result['rouge']['rougeL']
        bert_score = result['bertscore']['f1']
        if rouge_score > best_result[0] and bert_score > best_result[1]:
            best_result = (rouge_score, bert_score)
            best_step = step
    args.model = args.output_dir + "/round_" + str(best_step)
    test(args, logger, accelerator, tokenizer, dataset=dataset['test'])


def post_edit_evaluate(args, logger, accelerator):

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Prepare model with our `accelerator`.
    model = accelerator.prepare(model)

    # Load datasets.
    dataset = get_post_edit_training_datasets(args, accelerator, tokenizer,
        logger=logger if args.post_edit_output_raw_results else None,
        split_num=args.post_edit_split_num,
        cut_ratio=args.post_edit_cut_ratio,
        add_ratio=args.post_edit_add_ratio,
        keep_size=args.post_edit_keep_size)

    # evaluation
    test(args, logger, accelerator, tokenizer, dataset=dataset['validation'])
    test(args, logger, accelerator, tokenizer, dataset=dataset['test'])
