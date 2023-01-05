# bart large beam
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model facebook/bart-large \
    -log_file logs/tweetsumm_omission_bart_large.log \
    -output_dir models/tweetsumm_omission_bart_large \
    -dataset data/tweetsumm/tweetsumm \
    -max_source_length 1024 \
    -max_target_length 120 \
    -per_device_train_batch_size 4 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 1 \
    -val_max_target_length 120 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -save_path data/tweetsumm_omission/bart_large \
    -preprocessing_num_workers 1 \
    -result_dir results/tweetsumm

# bart base beam
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model facebook/bart-base \
    -log_file logs/tweetsumm_omission_bart_base.log \
    -output_dir models/tweetsumm_omission_bart_base \
    -dataset data/tweetsumm/tweetsumm \
    -max_source_length 1024 \
    -max_target_length 120 \
    -per_device_train_batch_size 8 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 1 \
    -val_max_target_length 120 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -save_path data/tweetsumm_omission/bart_base \
    -preprocessing_num_workers 1 \
    -result_dir results/tweetsumm


# baseline beam
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model facebook/bart-base \
    -baseline True \
    -log_file logs/tweetsumm_omission_baseline.log \
    -output_dir models/tweetsumm_omission_baseline \
    -dataset data/tweetsumm/tweetsumm \
    -max_source_length 1024 \
    -max_target_length 120 \
    -per_device_train_batch_size 4 \
    -per_device_eval_batch_size 4 \
    -learning_rate 1e-4 \
    -num_train_epochs 20 \
    -weight_decay 0.1 \
    -num_warmup_steps 1000 \
    -gradient_accumulation_steps 1 \ \
    -val_max_target_length 120 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -save_path data/tweetsumm_omission/baseline \
    -preprocessing_num_workers 1 \
    -result_dir results/tweetsumm


# bart large sample
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model facebook/bart-large \
    -log_file logs/tweetsumm_omission_bart_large.log \
    -output_dir models/tweetsumm_omission_bart_large \
    -dataset data/tweetsumm/tweetsumm \
    -max_source_length 1024 \
    -max_target_length 120 \
    -per_device_train_batch_size 8 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 1 \
    -val_max_target_length 120 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -do_sample True \
    -save_path data/tweetsumm_omission/bart_large \
    -preprocessing_num_workers 1 \
    -result_dir results/tweetsumm


# bart base sample
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model facebook/bart-base \
    -log_file logs/tweetsumm_omission_bart_base.log \
    -output_dir models/tweetsumm_omission_bart_base \
    -dataset data/tweetsumm/tweetsumm \
    -max_source_length 1024 \
    -max_target_length 120 \
    -per_device_train_batch_size 8 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 1 \
    -val_max_target_length 120 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -do_sample True \
    -save_path data/tweetsumm_omission/bart_base \
    -preprocessing_num_workers 1 \
    -result_dir results/tweetsumm


# baseline sample
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model facebook/bart-base \
    -baseline True \
    -log_file logs/tweetsumm_omission_baseline.log \
    -output_dir models/tweetsumm_omission_baseline \
    -dataset data/tweetsumm/tweetsumm \
    -max_source_length 1024 \
    -max_target_length 120 \
    -per_device_train_batch_size 4 \
    -per_device_eval_batch_size 4 \
    -learning_rate 1e-4 \
    -num_train_epochs 20 \
    -weight_decay 0.1 \
    -num_warmup_steps 1000 \
    -gradient_accumulation_steps 1 \ \
    -val_max_target_length 120 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -do_sample True \
    -save_path data/tweetsumm_omission/baseline \
    -preprocessing_num_workers 1 \
    -result_dir results/tweetsumm


# t5 base beam
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model t5-base \
    -log_file logs/tweetsumm_omission_t5_base.log \
    -output_dir models/tweetsumm_omission_t5_base \
    -dataset data/tweetsumm/tweetsumm \
    -prefix summarize: \
    -max_source_length 1024 \
    -max_target_length 120 \
    -per_device_train_batch_size 8 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 1 \
    -val_max_target_length 120 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -save_path data/tweetsumm_omission/t5_base \
    -preprocessing_num_workers 1 \
    -result_dir results/tweetsumm

# t5 small beam
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model t5-small \
    -log_file logs/tweetsumm_omission_t5_small.log \
    -output_dir models/tweetsumm_omission_t5_small \
    -dataset data/tweetsumm/tweetsumm \
    -prefix summarize: \
    -max_source_length 1024 \
    -max_target_length 120 \
    -per_device_train_batch_size 8 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 1 \
    -val_max_target_length 120 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -save_path data/tweetsumm_omission/t5_small \
    -preprocessing_num_workers 1 \
    -result_dir results/tweetsumm

# t5 base sample
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model t5-base \
    -log_file logs/tweetsumm_omission_t5_base.log \
    -output_dir models/tweetsumm_omission_t5_base \
    -dataset data/tweetsumm/tweetsumm \
    -prefix summarize: \
    -max_source_length 1024 \
    -max_target_length 120 \
    -per_device_train_batch_size 8 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 1 \
    -val_max_target_length 120 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -do_sample True \
    -save_path data/tweetsumm_omission/t5_base \
    -preprocessing_num_workers 1 \
    -result_dir results/tweetsumm

# t5 small sample
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model t5-small \
    -log_file logs/tweetsumm_omission_t5_small.log \
    -output_dir models/tweetsumm_omission_t5_small \
    -dataset data/tweetsumm/tweetsumm \
    -prefix summarize: \
    -max_source_length 1024 \
    -max_target_length 120 \
    -per_device_train_batch_size 8 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 1 \
    -val_max_target_length 120 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -do_sample True \
    -save_path data/tweetsumm_omission/t5_small \
    -preprocessing_num_workers 1 \
    -result_dir results/tweetsumm
