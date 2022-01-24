## 百度网盘AI大赛——图像处理挑战赛：文档图像摩尔纹消除第六名方案

- 训练代码
```
CUDA_VISIBLE_DEVICES=0 python train.py --exp_dir exp/asl_0.05_ms_cosine_no_local --dataset_train train_9.txt --dataset_val val_1.txt --lr 2e-4 --beta 0.05 --max_epoch 1500 --save_interval 40
```
训练log在exp_dir下，训练数据和验证数据划分为9：1，通过txt读取, 每40epoch验证一次。

- 测试代码
```
CUDA_VISIBLE_DEVICES=0 python predict.py --dataset_root ./moire_testB_dataset/ --pretrained model.pdparams
```

dataset_root 为 数据集路径，pretrained 为 模型参数路径

