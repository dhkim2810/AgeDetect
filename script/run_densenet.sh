python /root/volume/AgeDetect/main.py \
        --use_gpu --trial 4 --arch densenet \
        --optim sgd --learning_rate 1e-4 --nesterov --gamma 0.1 \
        --scheduler multi_step --milestone 300 420 470 \
        --epochs 500 --img_size 120 --da --batch_size 64
python /root/volume/AgeDetect/eval.py \
        --use_gpu --eval --trial 4 --arch densenet \
        --img_size 120