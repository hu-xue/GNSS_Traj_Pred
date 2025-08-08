set -e
python train.py --config_file config/train_img/klt3_train.json --bool_gnss --bool_fisheye --bool_surrounding --bool_ccffm --epoch 100 --lr 0.01