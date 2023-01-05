# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_train True \
    -seed 1 \
    -log_file logs/samsum_post_edit_train_bart.log \
    -output_dir models/post_edit_bart_large/samsum \
    -dataset data/samsum/omission.save \
    -model facebook/bart-large \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_train_batch_size 16 \
    -per_device_eval_batch_size 32 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 2 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -preprocessing_num_workers 16

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_train True \
    -seed 1 \
    -log_file logs/samsum_post_edit_train_t5.log \
    -output_dir models/post_edit_t5_small/samsum \
    -dataset data/samsum/omission.save \
    -model t5-small \
    -prefix summarize: \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_train_batch_size 32 \
    -per_device_eval_batch_size 32 \
    -learning_rate 3e-4 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 1 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -preprocessing_num_workers 16

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_train True \
    -seed 1 \
    -log_file logs/samsum_post_edit_train_bart_full_dial.log \
    -output_dir models/post_edit_bart_large_full_dial/samsum \
    -dataset data/samsum/omission.save \
    -model facebook/bart-large \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_train_batch_size 16 \
    -per_device_eval_batch_size 32 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 2 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_split_num 1 \
    -preprocessing_num_workers 16

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_train True \
    -seed 1 \
    -log_file logs/samsum_post_edit_train_t5_full_dial.log \
    -output_dir models/post_edit_t5_small_full_dial/samsum \
    -dataset data/samsum/omission.save \
    -model t5-small \
    -prefix summarize: \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_train_batch_size 32 \
    -per_device_eval_batch_size 32 \
    -learning_rate 3e-4 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 1 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_split_num 1 \
    -preprocessing_num_workers 16
