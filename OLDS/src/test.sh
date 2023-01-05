# samsum roberta-base test pair
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_pair/round_2 \
    -dataset data/samsum/omission.save \
    -domain all \
    -log_file logs_detection_test/samsum.roberta_base.pair.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode pair

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_pair/round_2 \
    -dataset data/samsum/omission.save \
    -domain bart_large \
    -log_file logs_detection_test/samsum.roberta_base.pair.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode pair

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_pair/round_2 \
    -dataset data/samsum/omission.save \
    -domain bart_base \
    -log_file logs_detection_test/samsum.roberta_base.pair.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode pair

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_pair/round_2 \
    -dataset data/samsum/omission.save \
    -domain t5_base \
    -log_file logs_detection_test/samsum.roberta_base.pair.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode pair

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_pair/round_2 \
    -dataset data/samsum/omission.save \
    -domain t5_small \
    -log_file logs_detection_test/samsum.roberta_base.pair.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode pair

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_pair/round_2 \
    -dataset data/samsum/omission.save \
    -domain baseline \
    -log_file logs_detection_test/samsum.roberta_base.pair.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode pair

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_pair/round_2 \
    -dataset data/samsum/omission.save \
    -domain pegasus \
    -log_file logs_detection_test/samsum.roberta_base.pair.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode pair

# samsum roberta-base test seq
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_seq/round_3 \
    -dataset data/samsum/omission.save \
    -domain all \
    -log_file logs_detection_test/samsum.roberta_base.seq.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode seq

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_seq/round_3 \
    -dataset data/samsum/omission.save \
    -domain bart_large \
    -log_file logs_detection_test/samsum.roberta_base.seq.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode seq

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_seq/round_3 \
    -dataset data/samsum/omission.save \
    -domain bart_base \
    -log_file logs_detection_test/samsum.roberta_base.seq.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode seq

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_seq/round_3 \
    -dataset data/samsum/omission.save \
    -domain t5_base \
    -log_file logs_detection_test/samsum.roberta_base.seq.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode seq

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_seq/round_3 \
    -dataset data/samsum/omission.save \
    -domain t5_small \
    -log_file logs_detection_test/samsum.roberta_base.seq.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode seq

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_seq/round_3 \
    -dataset data/samsum/omission.save \
    -domain baseline \
    -log_file logs_detection_test/samsum.roberta_base.seq.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode seq

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_seq/round_3 \
    -dataset data/samsum/omission.save \
    -domain pegasus \
    -log_file logs_detection_test/samsum.roberta_base.seq.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode seq

# samsum roberta-base test span
PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_span/round_5 \
    -dataset data/samsum/omission.save \
    -domain all \
    -log_file logs_detection_test/samsum.roberta_base.span.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode span

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_span/round_5 \
    -dataset data/samsum/omission.save \
    -domain bart_large \
    -log_file logs_detection_test/samsum.roberta_base.span.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode span

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_span/round_5 \
    -dataset data/samsum/omission.save \
    -domain bart_base \
    -log_file logs_detection_test/samsum.roberta_base.span.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode span

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_span/round_5 \
    -dataset data/samsum/omission.save \
    -domain t5_base \
    -log_file logs_detection_test/samsum.roberta_base.span.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode span

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_span/round_5 \
    -dataset data/samsum/omission.save \
    -domain t5_small \
    -log_file logs_detection_test/samsum.roberta_base.span.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode span

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_span/round_5 \
    -dataset data/samsum/omission.save \
    -domain baseline \
    -log_file logs_detection_test/samsum.roberta_base.span.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode span

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_test True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_span/round_5 \
    -dataset data/samsum/omission.save \
    -domain pegasus \
    -log_file logs_detection_test/samsum.roberta_base.span.test.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode span
