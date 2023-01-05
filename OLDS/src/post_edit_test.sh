# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_bart_large/samsum/round_1 \
    -log_file logs/samsum_post_edit_test_bart.log \
    -dataset data/samsum/omission.save \
    -domain all \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_output_raw_results True \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_bart_large_full_dial/samsum/round_1 \
    -log_file logs/samsum_post_edit_test_bart.log \
    -dataset data/samsum/omission.save \
    -domain all \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_split_num 1 \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_bart_large/samsum/round_1 \
    -log_file logs/samsum_post_edit_test_bart.log \
    -dataset data/samsum/omission.save \
    -domain bart_large \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_output_raw_results True \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_bart_large_full_dial/samsum/round_1 \
    -log_file logs/samsum_post_edit_test_bart.log \
    -dataset data/samsum/omission.save \
    -domain bart_large \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_split_num 1 \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_bart_large/samsum/round_1 \
    -log_file logs/samsum_post_edit_test_bart.log \
    -dataset data/samsum/omission.save \
    -domain bart_base \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_output_raw_results True \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_bart_large_full_dial/samsum/round_1 \
    -log_file logs/samsum_post_edit_test_bart.log \
    -dataset data/samsum/omission.save \
    -domain bart_base \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_split_num 1 \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_bart_large/samsum/round_1 \
    -log_file logs/samsum_post_edit_test_bart.log \
    -dataset data/samsum/omission.save \
    -domain t5_base \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_output_raw_results True \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_bart_large_full_dial/samsum/round_1 \
    -log_file logs/samsum_post_edit_test_bart.log \
    -dataset data/samsum/omission.save \
    -domain t5_base \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_split_num 1 \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_bart_large/samsum/round_1 \
    -log_file logs/samsum_post_edit_test_bart.log \
    -dataset data/samsum/omission.save \
    -domain t5_small \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_output_raw_results True \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_bart_large_full_dial/samsum/round_1 \
    -log_file logs/samsum_post_edit_test_bart.log \
    -dataset data/samsum/omission.save \
    -domain t5_small \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_split_num 1 \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_bart_large/samsum/round_1 \
    -log_file logs/samsum_post_edit_test_bart.log \
    -dataset data/samsum/omission.save \
    -domain baseline \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_output_raw_results True \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_bart_large_full_dial/samsum/round_1 \
    -log_file logs/samsum_post_edit_test_bart.log \
    -dataset data/samsum/omission.save \
    -domain baseline \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_split_num 1 \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_bart_large/samsum/round_1 \
    -log_file logs/samsum_post_edit_test_bart.log \
    -dataset data/samsum/omission.save \
    -domain pegasus \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_output_raw_results True \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_bart_large_full_dial/samsum/round_1 \
    -log_file logs/samsum_post_edit_test_bart.log \
    -dataset data/samsum/omission.save \
    -domain pegasus \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_split_num 1 \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_t5_small/samsum/round_5 \
    -log_file logs/samsum_post_edit_test_t5.log \
    -dataset data/samsum/omission.save \
    -domain all \
    -prefix summarize: \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_output_raw_results True \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_t5_small_full_dial/samsum/round_5 \
    -log_file logs/samsum_post_edit_test_t5.log \
    -dataset data/samsum/omission.save \
    -domain all \
    -prefix summarize: \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_split_num 1 \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_t5_small/samsum/round_5 \
    -log_file logs/samsum_post_edit_test_t5.log \
    -dataset data/samsum/omission.save \
    -domain bart_large \
    -prefix summarize: \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_output_raw_results True \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_t5_small_full_dial/samsum/round_5 \
    -log_file logs/samsum_post_edit_test_t5.log \
    -dataset data/samsum/omission.save \
    -domain bart_large \
    -prefix summarize: \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_split_num 1 \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_t5_small/samsum/round_5 \
    -log_file logs/samsum_post_edit_test_t5.log \
    -dataset data/samsum/omission.save \
    -domain bart_base \
    -prefix summarize: \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_output_raw_results True \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_t5_small_full_dial/samsum/round_5 \
    -log_file logs/samsum_post_edit_test_t5.log \
    -dataset data/samsum/omission.save \
    -domain bart_base \
    -prefix summarize: \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_split_num 1 \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_t5_small/samsum/round_5 \
    -log_file logs/samsum_post_edit_test_t5.log \
    -dataset data/samsum/omission.save \
    -domain t5_base \
    -prefix summarize: \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_output_raw_results True \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_t5_small_full_dial/samsum/round_5 \
    -log_file logs/samsum_post_edit_test_t5.log \
    -dataset data/samsum/omission.save \
    -domain t5_base \
    -prefix summarize: \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_split_num 1 \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_t5_small/samsum/round_5 \
    -log_file logs/samsum_post_edit_test_t5.log \
    -dataset data/samsum/omission.save \
    -domain t5_small \
    -prefix summarize: \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_output_raw_results True \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_t5_small_full_dial/samsum/round_5 \
    -log_file logs/samsum_post_edit_test_t5.log \
    -dataset data/samsum/omission.save \
    -domain t5_small \
    -prefix summarize: \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_split_num 1 \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_t5_small/samsum/round_5 \
    -log_file logs/samsum_post_edit_test_t5.log \
    -dataset data/samsum/omission.save \
    -domain baseline \
    -prefix summarize: \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_output_raw_results True \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_t5_small_full_dial/samsum/round_5 \
    -log_file logs/samsum_post_edit_test_t5.log \
    -dataset data/samsum/omission.save \
    -domain baseline \
    -prefix summarize: \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_split_num 1 \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_t5_small/samsum/round_5 \
    -log_file logs/samsum_post_edit_test_t5.log \
    -dataset data/samsum/omission.save \
    -domain pegasus \
    -prefix summarize: \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_output_raw_results True \
    -preprocessing_num_workers 1

# samsum
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_edit_test True \
    -seed 1 \
    -model models/post_edit_t5_small_full_dial/samsum/round_5 \
    -log_file logs/samsum_post_edit_test_t5.log \
    -dataset data/samsum/omission.save \
    -domain pegasus \
    -prefix summarize: \
    -max_source_length 512 \
    -max_target_length 90 \
    -per_device_eval_batch_size 32 \
    -val_max_target_length 90 \
    -val_min_target_length 1 \
    -num_beams 5 \
    -post_edit_split_num 1 \
    -preprocessing_num_workers 1
