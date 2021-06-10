python /root/volume/AgeDetect/rvc_main.py \
        --use_gpu --trial 21 --cudnn --arch dldlv2 \
        --optim adam --learning_rate 1e-3 \
        --scheduler step --step_size 30 --gamma 0.1 \
        --epochs 90 --train_val_ratio 0.9 \
        --da --thumbnail --img_size 224 \
        --batch_size 128
python /root/volume/AgeDetect/rvc_plot.py --arch dldlv2 --trial 21
python /root/volume/AgeDetect/rvc_main.py \
        --use_gpu --trial 22 --cudnn --arch dldlv2 \
        --optim adam --learning_rate 1e-3 \
        --scheduler step --step_size 30 --gamma 0.1 \
        --epochs 120 --train_val_ratio 0.9 \
        --da --thumbnail --img_size 224 \
        --batch_size 128
python /root/volume/AgeDetect/rvc_plot.py --arch dldlv2 --trial 22
python /root/volume/AgeDetect/rvc_main.py \
        --use_gpu --trial 23 --cudnn --arch dldlv2 \
        --optim adam --learning_rate 1e-3 \
        --scheduler step --step_size 30 --gamma 0.1 \
        --epochs 150 --train_val_ratio 0.9 \
        --da --thumbnail --img_size 224 \
        --batch_size 128
python /root/volume/AgeDetect/rvc_plot.py --arch dldlv2 --trial 23
python /root/volume/AgeDetect/rvc_main.py \
        --use_gpu --trial 24 --cudnn --arch dldlv2 \
        --optim adam --learning_rate 1e-3 \
        --scheduler step --step_size 30 --gamma 0.1 \
        --epochs 180 --train_val_ratio 0.9 \
        --da --thumbnail --img_size 224 \
        --batch_size 128
python /root/volume/AgeDetect/rvc_plot.py --arch dldlv2 --trial 24