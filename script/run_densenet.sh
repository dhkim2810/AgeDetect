python main.py --use_gpu --trial 1 --arch densenet \
                --da --use_huber --optim sgd --nesterov \
                --scheduler multi_step --milestone --60 90 120 \
                --workers 4