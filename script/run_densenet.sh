python /root/volume/AgeDetect/main.py --use_gpu --cudn --trial 1 --arch densenet \
                --use_huber --optim sgd --nesterov \
                --scheduler multi_step --milestone 60 90 120 \
                --workers 4 --batch_size 32 --learning_rate 1e-4