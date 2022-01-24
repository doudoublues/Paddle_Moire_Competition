import argparse
from operator import ge
import os.path
import random
import time
import datetime
import sys
import cv2
import numpy as np
import paddle
from PIL import Image
from paddle import nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from transforms import RandomHorizontalFlip, Resize, Normalize, Crop, RandomTranspose, RandomVerticalFlip, ColorJitter2, normalize
from dataset import Dataset
from model import MBCNN, MBCNN_RCAN, MBCNN_NL, MBCNN_CBAM
from losses import Sobel_loss, L1_Charbonnier_loss, MS_SSIM, PSNR
import logging
from utils import load_pretrained_model

import paddle.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    parser.add_argument(
        '--beta',
        dest='beta',
        help='beta',
        type=float,
        default=0.25
    )

    parser.add_argument(
        '--lr',
        dest='lr',
        help='learning rate',
        type=float,
        default=1e-4
    )

    parser.add_argument(
        '--step_size',
        dest='step_size',
        help='step_size',
        type=int,
        default=500
    )

    parser.add_argument(
        '--patch_size',
        dest='patch_size',
        help='patch_size',
        type=int,
        default=256
    )
    parser.add_argument(
        '--exp_dir',
        dest='exp_dir',
        help='exp_dir',
        type=str,
        default=None)

    parser.add_argument(
        '--dataset_train',
        dest='dataset_train',
        help='The path of dataset train',
        type=str,
        default=None)
    
    parser.add_argument(
        '--dataset_val',
        dest='dataset_val',
        help='The path of dataset val',
        type=str,
        default=None)


    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='batch_size',
        type=int,
        default=8
    )

    parser.add_argument(
        '--max_epochs',
        dest='max_epochs',
        help='max_epochs',
        type=int,
        default=700
    )


    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='log_iters',
        type=int,
        default=10
    )

    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='save_interval',
        type=int,
        default=20
    )

    parser.add_argument(
        '--sample_interval',
        dest='sample_interval',
        help='sample_interval',
        type=int,
        default=100
    )

    parser.add_argument(
        '--seed',
        dest='seed',
        help='random seed',
        type=int,
        default=1234
    )

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default=None)


    return parser.parse_args()


def sample_images(epoch, i, real_A, real_B, fake_B, exp_dir):
    data, pred, label = real_A * 255, fake_B * 255, real_B * 255
    pred = paddle.clip(pred.detach(), 0, 255)

    data = data.cast('int64')
    pred = pred.cast('int64')
    label = label.cast('int64')
    h, w = pred.shape[-2], pred.shape[-1]
    img = np.zeros((h, 1 * 3 * w, 3))
    for idx in range(0, 1):
        row = idx * h
        tmplist = [data[idx], pred[idx], label[idx]]
        for k in range(3):
            col = k * w
            tmp = np.transpose(tmplist[k], (1, 2, 0))
            img[row:row + h, col:col + w] = np.array(tmp)
    img = img.clip(0,255)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    img.save(os.path.join(exp_dir,"%03d_%06d.png" % (epoch, i)))


def main(args):

    # dist.init_parallel_env()

    paddle.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    

    # mean=(0.5643, 0.5572, 0.5426)
    # std=(0.2115, 0.2098, 0.2156)
    
    mean=(0, 0, 0)
    std=(1, 1, 1)

    exp_dir = args.exp_dir
    beta = args.beta
    train_transforms = [
        Crop(args.patch_size),
        # ColorJitter2(),
        RandomTranspose(),
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        Normalize(mean, std)
    ]
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    log_path = os.path.join(exp_dir, 'log.txt')
    if os.path.exists(log_path):
        os.remove(log_path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 

    fh = logging.FileHandler(log_path, 'w')
    fh.setLevel(logging.DEBUG) 
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    train_dataset = Dataset(dataset_root=args.dataset_train, transforms=train_transforms, mode="train")
    train_dataloader = paddle.io.DataLoader(train_dataset, batch_size=args.batch_size,
                                      num_workers=8, shuffle=True, drop_last=True,
                                      return_list=True)
    val_images_list= [path.strip() for path in open(args.dataset_val, 'r').readlines()]
    val_gts_list= [path.replace("images", "gts") for path in val_images_list]
    val_images_list.sort()
    val_gts_list.sort()
    best_score = 0
  
    # Loss functions
    # criterion_pixelwise = paddle.nn.L1Loss()  
    criterion_pixelwise = L1_Charbonnier_loss()
    criterion_ms_ssim = MS_SSIM(data_range=1.)
    criterion_psnr = PSNR(max_val=1.0)

    # model = MBCNN(64)
    # model = MBCNN_RCAN(64)
    model = MBCNN_NL(64)
    # model = MBCNN_CBAM(64)
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # model = paddle.DataParallel(model)
    # 获取参数情况
    for p in model.parameters():
        mulValue = np.prod(p.shape)  # 使用numpy prod接口计算数组所有元素之积
        Total_params += mulValue  # 总参数量
        if p.stop_gradient:
            NonTrainable_params += mulValue  # 可训练参数量
        else:
            Trainable_params += mulValue  # 非可训练参数量


    logger.info(f'Total params: {Total_params}')
    logger.info(f'Trainable params: {Trainable_params}')
    logger.info(f'Non-trainable params: {NonTrainable_params}')


    if args.pretrained is not None:
        load_pretrained_model(model, args.pretrained)

    # scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.lr, step_size=args.step_size, gamma=0.2, verbose=True)
    # scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=args.lr, T_max=40, verbose=True)
    # optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=scheduler)

    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=args.lr)

    prev_time = time.time()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for i, data_batch in enumerate(train_dataloader):
            inputs = data_batch[0]
            target_s1 = data_batch[1]
            target_s2 = data_batch[2]
            target_s3 = data_batch[3]
            
            output_s3, output_s2, output_s1 = model(inputs)

            l1_s1 = criterion_pixelwise(output_s1, target_s1)
            l1_s2 = criterion_pixelwise(output_s2, target_s2)
            l1_s3 = criterion_pixelwise(output_s3, target_s3)

            sobel_s1 = Sobel_loss(output_s1, target_s1)
            sobel_s2 = Sobel_loss(output_s2, target_s2)
            sobel_s3 = Sobel_loss(output_s3, target_s3)

            ms_s1 = criterion_ms_ssim(output_s1, target_s1)
            ms_s2 = criterion_ms_ssim(output_s2, target_s2)
            ms_s3 = criterion_ms_ssim(output_s3, target_s3)

            pixel_loss = l1_s1 + l1_s2 + l1_s3
            sobel_loss = sobel_s1 + sobel_s2 + sobel_s3
            ms_loss = ms_s1 + ms_s2 + ms_s3

            loss = pixel_loss + beta * sobel_loss + ms_loss
        
            
            loss.backward()

            optimizer.step()
            optimizer.clear_gradients()

            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + i
            batches_left = args.max_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            if i % args.log_iters == 0:
                sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [loss: %f  pixel loss: %f sobel loss: %f, ms_loss: %f] ETA: %s" %
                                 (epoch, args.max_epochs,
                                  i, len(train_dataloader),
                                  loss,
                                  pixel_loss,
                                  sobel_loss,
                                  ms_loss,
                                  time_left))
                logger.info("\r[Epoch %d/%d] [Batch %d/%d] [loss: %f  pixel loss: %f sobel loss: %f, ms_loss: %f] ETA: %s" %
                                 (epoch, args.max_epochs,
                                  i, len(train_dataloader),
                                  loss,
                                  pixel_loss,
                                  sobel_loss,
                                  ms_loss,
                                  time_left))

            if i % args.sample_interval == 0:
                sample_images(epoch, i, inputs, target_s1, output_s1, exp_dir)
        # scheduler.step()
        if epoch % args.save_interval == 0:
            model.eval()
            psnr = 0
            ssim = 0
            with paddle.no_grad():
                for i in range(len(val_images_list)):

                    img = cv2.imread(val_images_list[i])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, c = img.shape
                    h_pad = 0
                    w_pad = 0

                    if h % 8 != 0:
                        h_pad = (h // 8 + 1) * 8 - h
                    if w % 8 != 0:
                        w_pad = (w // 8 + 1) * 8 - w
                    img = np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)))

                    gt = cv2.imread(val_gts_list[i])
                    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                    
                    img = normalize(img, mean, std)
                    img = paddle.to_tensor(img)
                    # img /= 255.0
                    img = paddle.transpose(img, [2, 0, 1])
                    img = img.unsqueeze(0)

                    gt = paddle.to_tensor(gt)
                    gt /= 255.0
                    gt = paddle.transpose(gt, [2, 0, 1])
                    gt = gt.unsqueeze(0)


                    img_out = model(img)[-1]

                    # img_out = img_out.squeeze(0)
                    img_out = img_out[:, :, :h, :w].clip(0.0, 1.0)

                    psnr += criterion_psnr(img_out, gt)
                    ssim += (1 - criterion_ms_ssim(img_out, gt)).numpy()[0]

                    # img_out = img_out * 255.0
                    # img_out = paddle.clip(img_out, 0, 255)
                    # img_out = paddle.transpose(img_out, [1, 2, 0])
                    # img_out = img_out.numpy().round().clip(0,255).astype(np.uint8)

                    # psnr += peak_signal_noise_ratio(img_out, gt)
                    # ssim += structural_similarity(img_out,gt,multichannel=True)
            psnr /= len(val_images_list)
            ssim /= len(val_images_list)
            score = 0.5 * psnr / 100 + 0.5 * ssim

            if score > best_score:
                best_score = score
                best_save_dir =  os.path.join(exp_dir, "model", 'best_model')
                if not os.path.exists(best_save_dir):
                    os.makedirs(best_save_dir)
                paddle.save(model.state_dict(),
                    os.path.join(best_save_dir, 'model.pdparams'))
                paddle.save(optimizer.state_dict(),
                            os.path.join(best_save_dir, 'model.pdopt'))
            print("\r[Validation] [Epoch %d/%d] [psnr: %f, ssim: %f, score: %f, best score: %f] " %
                                (epoch, args.max_epochs,
                                  psnr,
                                  ssim,
                                  score,
                                  best_score))
            logger.info("\r[Validation] [Epoch %d/%d] [psnr: %f, ssim: %f, score: %f, best score: %f]" %
                                (epoch, args.max_epochs,
                                  psnr,
                                  ssim,
                                  score,
                                  best_score))

            current_save_dir = os.path.join(exp_dir, "model", f'epoch_{epoch}')
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)
            paddle.save(model.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(current_save_dir, 'model.pdopt'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
    # dist.spawn(main, args, nprocs=3, gpus='2,3,4')
