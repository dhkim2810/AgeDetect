python /root/volume/AgeDetect/main.py \
        --use_gpu --cudnn --trial 16 --arch spinalresnet18 \
        --optim sgd --learning_rate 1e-4 --nesterov \
        --scheduler multi_step --milestone 240 320 380
        --epochs 400 --img_size 100 --da
python /root/volume/AgeDetect/eval.py \
        --use_gpu --eval --trial 16 --arch spinalresnet18 \
        --img_size 100
python /root/volume/AgeDetect/plot.py --arch spinalresnet18 --trial 16