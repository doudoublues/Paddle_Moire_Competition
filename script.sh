# train
CUDA_VISIBLE_DEVICES=0 python train.py --exp_dir exp/asl_0.05_ms_cosine_no_local --dataset_train train_9.txt --dataset_val val_1.txt --lr 2e-4 --beta 0.05 --max_epoch 1500 --save_interval 40

# test
CUDA_VISIBLE_DEVICES=0 python predict.py --dataset_root ./moire_testB_dataset/ --pretrained model.pdparams