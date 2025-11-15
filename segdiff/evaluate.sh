test_data_dir="/root/autodl-tmp/data/BraTS2021/processed_data"

pred_data_dir='/root/autodl-fs/Diffusion-based-Segmentation-main/outputs'

data_name='BRATS'

python scripts/segmentation_evaluate.py \
    --test_data_dir $test_data_dir \
    --pred_data_dir $pred_data_dir \
    --image_size 256 \
    --data_name $data_name 
    