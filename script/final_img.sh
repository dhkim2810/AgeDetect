python /root/volume/AgeDetect/rvc_main.py \
        --use_gpu --trial 11 --cudnn --arch dldlv2 \
        --optim adam --learning_rate 1e-3 \
        --scheduler step --step_size 30 --gamma 0.1 \
        --epochs 90 --train_val_ratio 0.9 \
        --da --thumbnail --img_size 128 \
        --batch_size 128
python /root/volume/AgeDetect/rvc_plot.py --arch dldlv2 --trial 11
python /root/volume/AgeDetect/rvc_main.py \
        --use_gpu --trial 12 --cudnn --arch dldlv2 \
        --optim adam --learning_rate 1e-3 \
        --scheduler step --step_size 30 --gamma 0.1 \
        --epochs 90 --train_val_ratio 0.9 \
        --da --thumbnail --img_size 160 \
        --batch_size 128
python /root/volume/AgeDetect/rvc_plot.py --arch dldlv2 --trial 12
python /root/volume/AgeDetect/rvc_main.py \
        --use_gpu --trial 13 --cudnn --arch dldlv2 \
        --optim adam --learning_rate 1e-3 \
        --scheduler step --step_size 30 --gamma 0.1 \
        --epochs 90 --train_val_ratio 0.9 \
        --da --thumbnail --img_size 192 \
        --batch_size 128
python /root/volume/AgeDetect/rvc_plot.py --arch dldlv2 --trial 13
python /root/volume/AgeDetect/rvc_main.py \
        --use_gpu --trial 14 --cudnn --arch dldlv2 \
        --optim adam --learning_rate 1e-3 \
        --scheduler step --step_size 30 --gamma 0.1 \
        --epochs 90 --train_val_ratio 0.9 \
        --da --thumbnail --img_size 224 \
        --batch_size 128
python /root/volume/AgeDetect/rvc_plot.py --arch dldlv2 --trial 14