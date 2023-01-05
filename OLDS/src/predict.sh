# samsum bert-base test pair

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_predict True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_pair/round_3 \
    -dataset data/samsum/omission.save \
    -domain all \
    -log_file logs/samsum.roberta_base.pair.predict.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode pair

# samsum bert-base test seq

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_predict True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_seq/round_3 \
    -dataset data/samsum/omission.save \
    -domain all \
    -log_file logs/samsum.roberta_base.seq.predict.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode seq

# samsum bert-base test span

PYTHONPATH=.. accelerate launch ./src/main.py \
    -do_predict True \
    -model roberta-base \
    -checkpoint models/samsum_roberta_base_span/round_3 \
    -dataset data/samsum/omission.save \
    -domain all \
    -log_file logs/samsum.roberta_base.span.predict.log \
    -max_source_length 512 \
    -per_device_eval_batch_size 64 \
    -preprocessing_num_workers 16 \
    -mode span
