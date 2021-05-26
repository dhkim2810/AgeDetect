python /root/volume/AgeDetect/main.py \
        --use_gpu --cudnn --trial 14 --arch spinalresnet18 \
        --optim sgd --learning_rate 2e-4 --nesterov \
        --scheduler step --step_size 50 --gamma 0.1 \
        --epochs 300 --img_size 120 --da
python /root/volume/AgeDetect/eval.py \
        --use_gpu --eval --trial 14 --arch spinalresnet18 \
        --img_size 120
python /root/volume/AgeDetect/plot.py --arch spinalresnet18 --trial 14