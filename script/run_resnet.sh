python main.py --use_gpu --trial 1 --arch resnet18 \
                --optim sgd --nesterov \
                --scheduler multi_step --milestone 60 90 120 --workers 4  --img_size 64
python eval.py --use_gpu --eval --trial 1 --arch resnet18
python plot.py --arch resnet18 --trial 1