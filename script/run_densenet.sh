python main.py \
        --use_gpu --trial 1 --arch densenet \
        --optim adam --learning_rate 1e-4 --beta1 0.9 --beta2 0.95 \
        --scheduler multi_step --milestone 300 400 450 \
        --epochs 500 --img_size 144 --da --batch_size 64
python eval.py \
        --eval --use_gpu --trial 1 --arch densenet  --img_size 144
python plot.py --arch densenet --trial 1