python /root/volume/AgeDetect/rvc_main.py \
        --use_gpu --trial 13 --cudnn --arch random_bin --M 8 --N 29 \
        --optim adam --learning_rate 2e-4 \
        --scheduler multi_step --milestone 60 90 --gamma 0.1 \
        --epochs 120 --img_size 120 --da --batch_size 64 \
        --train_val_ratio 0.99 --thumbnail
python /root/volume/AgeDetect/rvc_eval.py \
        --eval --use_gpu --trial 13 --cudnn --arch random_bin \
        --img_size 80
python /root/volume/AgeDetect/rvc_plot.py --arch random_bin --trial 13