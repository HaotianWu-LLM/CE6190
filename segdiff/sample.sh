model_path='/root/autodl-fs/Diffusion-based-Segmentation-main/results/savedmodel010000.pt'

python scripts/modify_sample.py  \
    --data_dir '/root/autodl-tmp/data/BraTS2021/processed_data'  \
    --model_path $model_path \
    --num_ensemble 3 \
    --image_size 256 \
    --num_channels 128 \
    --class_cond False \
    --num_res_blocks 2 \
    --num_heads 1 \
    --learn_sigma True \
    --use_scale_shift_norm False \
    --attention_resolutions 16 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --rescale_learned_sigmas False \
    --rescale_timesteps False 
