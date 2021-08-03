python main.py \
        --use_gpu --trial 1 --cudnn --arch random_bin --M 10 --N 30 \
        --optim adam --learning_rate 2e-4 \
        --scheduler multi_step --milestone 60 90 --gamma 0.1 \
        --epochs 120 --img_size 120 --da --batch_size 64 \
        --train_val_ratio 0.99 --thumbnail
python eval.py --eval --use_gpu --trial 1 --cudnn --arch random_bin --img_size 120
python plot.py --arch random_bin --trial 1