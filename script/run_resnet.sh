python /root/volume/AgeDetect/main.py --use_gpu --trial 1 --arch resnet18 \
                --da --optim sgd --nesterov \
                --scheduler multi_step --milestone 60 90 120 \
                --workers 4