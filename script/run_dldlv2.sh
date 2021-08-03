python /root/volume/AgeDetect/main.py \
        --use_gpu --trial 1 --cudnn --arch dldlv2 \
        --optim adam --learning_rate 1e-3 \
        --scheduler multi_step --milestone 30 80 120 --gamma 0.1 \
        --epochs 150 --img_size 224 --da --batch_size 128 \
        --train_val_ratio 0.99 --thumbnail
python /root/volume/AgeDetect/eval.py --eval --use_gpu --trial 1 --cudnn --arch dldlv2 --img_size 224
python /root/volume/AgeDetect/plot.py --arch dldlv2 --trial 1