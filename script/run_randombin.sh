python /root/volume/AgeDetect/rvc_main.py \
        --use_gpu --trial 4 --cudnn --arch random_bin \
        --optim adam \
        --scheduler multi_step --milestone 180 240 280 --gamma 0.1 \
        --epochs 300 --img_size 224 --da --thumbnail --batch_size 64
python /root/volume/AgeDetect/rvc_eval.py \
        --eval --use_gpu --trial 4 --cudnn --arch random_bin \
        --img_size 224
python /root/volume/AgeDetect/rvc_plot.py --arch random_bin --trial 4