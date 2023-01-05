# samsum beam
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model google/pegasus-large \
    -log_file logs_omission_data/samsum_omission_pegasus.log \
    -output_dir models/samsum_omission_pegasus \
    -dataset samsum \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_train_batch_size 8 \
    -per_device_eval_batch_size 16 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 4 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -save_path data/samsum_omission/pegasus \
    -preprocessing_num_workers 32 \
    -result_dir results/samsum

# samsum sample
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model google/pegasus-large \
    -log_file logs_omission_data/samsum_omission_pegasus.log \
    -output_dir models/samsum_omission_pegasus \
    -dataset samsum \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_train_batch_size 8 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 4 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -do_sample True \
    -save_path data/samsum_omission/pegasus \
    -preprocessing_num_workers 32 \
    -result_dir results/samsum

# dialsumm beam
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model google/pegasus-large \
    -log_file logs_omission_data/dialsumm_omission_pegasus.log \
    -output_dir models/dialsumm_omission_pegasus \
    -dataset data/dialsumm/dialsumm \
    -max_source_length 512 \
    -max_target_length 150 \
    -per_device_train_batch_size 4 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 8 \
    -val_max_target_length 150 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -save_path data/dialsumm_omission/pegasus \
    -preprocessing_num_workers 32 \
    -result_dir results/dialsumm

# dialsumm sample
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model google/pegasus-large \
    -log_file logs_omission_data/dialsumm_omission_pegasus.log \
    -output_dir models/dialsumm_omission_pegasus \
    -dataset data/dialsumm/dialsumm \
    -max_source_length 512 \
    -max_target_length 150 \
    -per_device_train_batch_size 4 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 8 \
    -val_max_target_length 150 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -do_sample True \
    -save_path data/dialsumm_omission/pegasus \
    -preprocessing_num_workers 32 \
    -result_dir results/dialsumm

# tweetsumm beam
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model google/pegasus-large \
    -log_file logs_omission_data/tweetsumm_omission_pegasus.log \
    -output_dir models/tweetsumm_omission_pegasus \
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
    -save_path data/tweetsumm_omission/pegasus \
    -preprocessing_num_workers 1 \
    -result_dir results/tweetsumm

# tweetsumm sample
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model google/pegasus-large \
    -log_file logs_omission_data/tweetsumm_omission_pegasus.log \
    -output_dir models/tweetsumm_omission_pegasus \
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
    -do_sample True \
    -save_path data/tweetsumm_omission/pegasus \
    -preprocessing_num_workers 1 \
    -result_dir results/tweetsumm

# emailsum short beam
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model google/pegasus-large \
    -log_file logs_omission_data/emailsum_short_omission_pegasus.log \
    -output_dir models/emailsum_short_omission_pegasus \
    -dataset data/emailsum_short/emailsum_short \
    -max_source_length 1024 \
    -max_target_length 65 \
    -per_device_train_batch_size 2 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 2 \
    -val_max_target_length 65 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -save_path data/emailsum_short_omission/pegasus \
    -preprocessing_num_workers 1 \
    -result_dir results/emailsum_short

# emailsum short sample
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model google/pegasus-large \
    -log_file logs_omission_data/emailsum_short_omission_pegasus.log \
    -output_dir models/emailsum_short_omission_pegasus \
    -dataset data/emailsum_short/emailsum_short \
    -max_source_length 1024 \
    -max_target_length 65 \
    -per_device_train_batch_size 4 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 1 \
    -val_max_target_length 65 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -do_sample True \
    -save_path data/emailsum_short_omission/pegasus \
    -preprocessing_num_workers 1 \
    -result_dir results/emailsum_short

# emailsum long beam
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model google/pegasus-large \
    -log_file logs_omission_data/emailsum_long_omission_pegasus.log \
    -output_dir models/emailsum_long_omission_pegasus \
    -dataset data/emailsum_long/emailsum_long \
    -max_source_length 1024 \
    -max_target_length 160 \
    -per_device_train_batch_size 2 \
    -per_device_eval_batch_size 4 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 2 \
    -val_max_target_length 160 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -save_path data/emailsum_long_omission/pegasus \
    -preprocessing_num_workers 1 \
    -result_dir results/emailsum_long

# emailsum long sample
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model google/pegasus-large \
    -log_file logs_omission_data/emailsum_long_omission_pegasus.log \
    -output_dir models/emailsum_long_omission_pegasus \
    -dataset data/emailsum_long/emailsum_long \
    -max_source_length 1024 \
    -max_target_length 160 \
    -per_device_train_batch_size 4 \
    -per_device_eval_batch_size 2 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 1 \
    -val_max_target_length 160 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -do_sample True \
    -save_path data/emailsum_long_omission/pegasus \
    -preprocessing_num_workers 1 \
    -result_dir results/emailsum_long

# qmsum beam
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model google/pegasus-large \
    -log_file logs_omission_data/qmsum_omission_pegasus.log \
    -output_dir models/qmsum_omission_pegasus \
    -dataset data/qmsum/qmsum \
    -max_source_length 2048 \
    -max_target_length 200 \
    -per_device_train_batch_size 1 \
    -per_device_eval_batch_size 1 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 16 \
    -val_max_target_length 200 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -save_path data/qmsum_omission/pegasus \
    -preprocessing_num_workers 1 \
    -result_dir results/qmsum

# qmsum sample
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_process_train True \
    -seed 1 \
    -model google/pegasus-large \
    -log_file logs_omission_data/qmsum_omission_pegasus.log \
    -output_dir models/qmsum_omission_pegasus \
    -dataset data/qmsum/qmsum \
    -max_source_length 2048 \
    -max_target_length 200 \
    -per_device_train_batch_size 1 \
    -per_device_eval_batch_size 1 \
    -learning_rate 5e-5 \
    -num_train_epochs 5 \
    -weight_decay 0.1 \
    -num_warmup_steps 0 \
    -gradient_accumulation_steps 16 \
    -val_max_target_length 200 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -do_sample True \
    -save_path data/qmsum_omission/pegasus \
    -preprocessing_num_workers 1 \
    -result_dir results/qmsum