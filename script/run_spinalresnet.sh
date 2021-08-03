python main.py \
        --use_gpu --trial 1 --arch spinalresnet18 \
        --optim adam --learning_rate 1e-4 --beta1 0.5 \
        --scheduler multi_step --milestone 240 \
        --epochs 300 --img_size 200
python eval.py --use_gpu --eval --arch spinalresnet18 --trial 1
python plot.py --arch spinalresnet18 --trial 1