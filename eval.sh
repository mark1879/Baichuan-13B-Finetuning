CUDA_VISIBLE_DEVICES=0 python finetune_lora.py \
    --do_eval \
    --model_name_or_path baichuan-inc/Baichuan-13B-Chat \
    --checkpoint_dir baichuan_lora_checkpoint \
    --dataset_dir data \
    --dataset alpaca_gpt4_zh_test \
    --output_dir baichuan_lora_eval_result \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --padding_side right