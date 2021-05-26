python /root/volume/AgeDetect/main.py \
    --use_gpu --cudnn --trial 1 \
    --arch resnet18 \
    --workers 4 --batch_size 128 --epoch 100 \
    --optim sgd --nesterov --learning_rate 1e-4 \
    --scheduler fixed
python /root/volume/AgeDetect/eval.py --use_gpu --eval --trial 1 --arch resnet18
