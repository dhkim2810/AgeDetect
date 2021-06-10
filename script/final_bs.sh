python /root/volume/AgeDetect/rvc_main.py \
        --use_gpu --trial 31 --cudnn --arch dldlv2 \
        --optim adam --learning_rate 1e-3 \
        --scheduler step --step_size 30 --gamma 0.1 \
        --epochs 90 --train_val_ratio 0.9 \
        --da --thumbnail --img_size 224 \
        --batch_size 128
python /root/volume/AgeDetect/rvc_plot.py --arch dldlv2 --trial 31
python /root/volume/AgeDetect/rvc_main.py \
        --use_gpu --trial 32 --cudnn --arch dldlv2 \
        --optim adam --learning_rate 1e-3 \
        --scheduler step --step_size 30 --gamma 0.1 \
        --epochs 90 --train_val_ratio 0.9 \
        --da --thumbnail --img_size 224 \
        --batch_size 64
python /root/volume/AgeDetect/rvc_plot.py --arch dldlv2 --trial 32
python /root/volume/AgeDetect/rvc_main.py \
        --use_gpu --trial 33 --cudnn --arch dldlv2 \
        --optim adam --learning_rate 1e-3 \
        --scheduler step --step_size 30 --gamma 0.1 \
        --epochs 90 --train_val_ratio 0.9 \
        --da --thumbnail --img_size 224 \
        --batch_size 32
python /root/volume/AgeDetect/rvc_plot.py --arch dldlv2 --trial 33