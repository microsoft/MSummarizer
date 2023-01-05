import os
import math
import random
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from models.test import test
from models.detection import DetectionModel
from others.dataset import get_omission_datasets
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import WEIGHTS_NAME
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    DataCollatorForSeq2Seq,
    get_scheduler,
)


def train(args, logger, accelerator, tokenizer=None, model=None, dataset=None, save_final=False):

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    accelerator.free_memory()

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    if model is None:
        
        # model = AutoModelForSequenceClassification.from_pretrained(args.model)
        model = DetectionModel(args)

        # Prepare model with our `accelerator`.
        model = accelerator.prepare(model)

    if dataset is None:
        # Load datasets.
        dataset = get_omission_datasets(args, accelerator, tokenizer)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    train_dataset, eval_dataset = dataset['train'], dataset['validation']
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

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

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]

    if (args.mode == 'seq' or args.mode == 'pair') and 'large' in args.model:
        special_symbol = 'classifier'
    else:
        special_symbol = 'pn'

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and special_symbol not in n],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and special_symbol not in n],
            "weight_decay": 0.0,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and special_symbol in n],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate * 10,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and special_symbol in n],
            "weight_decay": 0.0,
            "lr": args.learning_rate * 10,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

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
            os.makedirs(output_dir, exist_ok=True)
            if isinstance(unwrapped_model, PreTrainedModel):
                unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            else:
                if accelerator.is_main_process:
                    state_dict = unwrapped_model.state_dict()
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
                # Good practice: save your training arguments together with the trained model
                torch.save(args, os.path.join(output_dir, "training_args.bin"))

    results = dict()
    for epoch in range(num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch, mode=args.mode, cls_token_id=tokenizer.cls_token_id)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                if accelerator.is_local_main_process and completed_steps % 10 == 0:
                    logger.info("Loss at step %d: %.4f" % (completed_steps, outputs.loss.item()))
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
