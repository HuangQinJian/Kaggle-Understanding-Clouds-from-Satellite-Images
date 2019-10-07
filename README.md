CUDA_VISIBLE_DEVICES=3,4 python train.py --train_csv_path data/train.csv --epoch 30 --log_dir logs --batch_size 8

CUDA_VISIBLE_DEVICES=0 python predict.py --weights_path logs/checkpoints/best.pth --batch_size 8