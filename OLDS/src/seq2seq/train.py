import os
import math
import random
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from seq2seq.test import test
from others.dataset import get_datasets
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    BartForConditionalGeneration,
    PegasusForConditionalGeneration,
    get_scheduler
)


def resize_position_embeddings(model, max_len):
    model_len = model.config.max_position_embeddings
    # encoder position resize
    pos_embed_encoder = BartLearnedPositionalEmbedding(max_len, model.config.d_model)
    pos_embed_encoder.weight.data[:model_len+2] = model.encoder.embed_positions.weight.data
    pos_embed_encoder.weight.data[model_len+2:] = model.encoder.embed_positions.weight.data[-1][None, :].repeat(max_len-model_len, 1)
    model.encoder.embed_positions = pos_embed_encoder
    
    # decoder position resize
    pos_embed_decoder = BartLearnedPositionalEmbedding(max_len, model.config.d_model)
    pos_embed_decoder.weight.data[:model_len+2] = model.decoder.embed_positions.weight.data
    pos_embed_decoder.weight.data[model_len+2:] = model.decoder.embed_positions.weight.data[-1][None, :].repeat(max_len-model_len, 1)
    model.decoder.embed_positions = pos_embed_decoder


def train(args, logger, accelerator, tokenizer=None, model=None, dataset=None, save_final=False):

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    accelerator.free_memory()

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    if model is None:
        if args.baseline:
            config = AutoConfig.from_pretrained(
                args.model,
                encoder_layers=6,
                encoder_ffn_dim=1024,
                encoder_attention_heads=8,
                decoder_layers=6,
                decoder_ffn_dim=1024,
                decoder_attention_heads=8,
                d_model=768,
                num_labels=args.num_labels,
                max_position_embeddings=args.max_source_length
            )
            model = AutoModelForSeq2SeqLM.from_config(config)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
            if isinstance(model, PegasusForConditionalGeneration) and \
                args.max_source_length > model.config.max_position_embeddings:
                model.resize_position_embeddings(args.max_source_length)
            elif isinstance(model, BartForConditionalGeneration) and \
                args.max_source_length > model.config.max_position_embeddings:
                resize_position_embeddings(model.model, args.max_source_length)
                model.config.max_position_embeddings = args.max_source_length
                    
        model.resize_token_embeddings(len(tokenizer))
        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        # Prepare model with our `accelerator`.
        model = accelerator.prepare(model)

    if dataset is None:
        # Load datasets.
        dataset = get_datasets(args, accelerator, tokenizer)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    train_dataset, eval_dataset = dataset['train'], dataset['validation']
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    optimizer, train_dataloader = accelerator.prepare(
        optimizer, train_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        num_train_epochs = args.num_train_epochs
    else:
        max_train_steps = args.max_train_steps
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    def model_save(model, tokenizer, accelerator, idx):
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            output_dir = args.output_dir + '/round_' + str(idx)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)

    results = dict()
    for epoch in range(num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if args.save_strategy == 'step' and \
                        (completed_steps % args.save_steps == 0 or completed_steps >= max_train_steps):
                    result = test(args, logger, accelerator, tokenizer, model, eval_dataset)
                    results[completed_steps] = result
                    if not save_final:
                        model_save(model, tokenizer, accelerator, completed_steps)
                    model.train()

            if completed_steps >= max_train_steps:
                break

        if args.save_strategy == 'epoch':
            result = test(args, logger, accelerator, tokenizer, model, eval_dataset)
            results[epoch+1] = result
            if not save_final:
                model_save(model, tokenizer, accelerator, epoch+1)
    if save_final:
        model_save(model, tokenizer, accelerator, 'final')

    return results
