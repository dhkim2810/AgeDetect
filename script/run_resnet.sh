python /root/volume/AgeDetect/main.py --use_gpu --trial 1 --arch resnet18 \
                --optim sgd --nesterov \
                --scheduler multi_step --milestone 60 90 120 --workers 4
python /root/volume/AgeDetect/eval.py --use_gpu --eval --trial 1 --arch resnet18