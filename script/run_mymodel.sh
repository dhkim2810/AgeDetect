python /root/volume/AgeDetect/main.py --use_gpu --cudnn --trial 2 --arch mymodel \
                --optim sgd --nesterov --learning_rate 1e-4 \
                --scheduler multi_step --milestone 60 90 120 \
                --workers 4 --batch_size 128
python /root/volume/AgeDetect/eval.py --use_gpu --eval --trial 2 --arch mymodel
