python /root/volume/AgeDetect/main.py \
        --use_gpu --trial 6 --arch densenet \
        --optim sgd --learning_rate 1e-4 --nesterov --gamma 0.1 \
        --scheduler multi_step --milestone 480 600 720 \
        --epochs 800 --img_size 120 --da --batch_size 64 --use_l1_loss
python /root/volume/AgeDetect/eval.py \
        --use_gpu --eval --trial 6 --arch densenet \
        --img_size 120