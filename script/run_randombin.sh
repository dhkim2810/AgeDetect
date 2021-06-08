python /root/volume/AgeDetect/rvc_main.py \
        --use_gpu --trial 3 --cudnn --arch random_bin \
        --optim adam \
        --scheduler multi_step --milestone 180 240 280 \
        --epochs 300 --img_size 224 --da --batch_size 64