## No Injection Canary
python3 run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --dataset_name macdial \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ./tmp \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    # --do_predict \
    
## Injection Canary Only X
python3 run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --dataset_name macdial \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ./tmp \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --add_canary True \
    --position_canary x \
    --canary_len 6 \
    --canary_rep 1000 \
    --do_train \
    --do_eval \
    # --do_predict \

## Injection Canary Only Y
python3 run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --dataset_name macdial \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ./tmp \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --add_canary True \
    --position_canary y \
    --canary_len 6 \ 
    --canary_rep 1000 \ 
    --do_train \
    --do_eval \
    # --do_predict \

# Injection Canary BOTH X and Y
python3 run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --dataset_name macdial \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ./tmp \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --add_canary True \
    --position_canary xy \
    --canary_len 6 \
    --canary_rep 1000 \ 
    --do_train \
    --do_eval \
    # --do_predict \