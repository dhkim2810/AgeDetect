python /root/volume/AgeDetect/rvc_main.py \
        --use_gpu --trial 7 --cudnn --arch random_bin \
        --optim adam --learning_rate 2e-4 \
        --scheduler multi_step --milestone 70 150 240 280 --gamma 0.5 \
        --epochs 300 --img_size 224 --da --batch_size 128
python /root/volume/AgeDetect/rvc_eval.py \
        --eval --use_gpu --trial 7 --cudnn --arch random_bin \
        --img_size 224
python /root/volume/AgeDetect/rvc_plot.py --arch random_bin --trial 7