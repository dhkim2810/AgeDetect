python /root/volume/AgeDetect/main.py \
        --use_gpu --trial 4 --arch spinalresnet18 \
        --optim adam --learning_rate 1e-4 --beta1 0.5 \
        --scheduler multi_step --milestone 240 \
        --epochs 300 --img_size 200
python /root/volume/AgeDetect/eval.py --use_gpu --eval --trial 4 --arch spinalresnet18