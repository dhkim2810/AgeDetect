python /root/volume/AgeDetect/main.py --use_gpu --cudnn --trial 2 --arch densenet \
                --optim sgd --nesterov \
                --scheduler multi_step --milestone 60 90 120 \
                --workers 4 --batch_size 64 --learning_rate 2e-4
python /root/volume/AgeDetect/eval.py --use_gpu --eval --trial 2 --arch densenet
