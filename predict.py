import argparse
from enum import Flag
import glob
import os.path
import paddle
import paddle.nn as nn
import cv2
import time
from model import MBCNN_NL
from utils import load_pretrained_model
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Model testing')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default='/Users/alex/baidu/mmaction2/data/ucf101/')

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default=None)

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='batch_size',
        type=int,
        default=32
    )

    return parser.parse_args()

def main(args):

    model = MBCNN_NL(64,False)

    if args.pretrained is not None:
        load_pretrained_model(model, args.pretrained)

    im_files = sorted(glob.glob(os.path.join(args.dataset_root, "*.jpg")))
    
    hor = [False, True]
    ver = [False, True]
    trans = [False, True]
    total_time = 0
    with paddle.no_grad():
        for i, im in enumerate(im_files):
            print(im)
            img = cv2.imread(im)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = np.zeros(img.shape, dtype='float32')
            height, width, c = img.shape
            
            start_time = time.time()
            h_pad = 0
            w_pad = 0

            if height % 8 != 0:
                h_pad = (height // 8 + 1) * 8 - height
            if width % 8 != 0:
                w_pad = (width // 8 + 1) * 8 - width

            img = np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), "reflect")
            print(img.shape)
            for h in hor:
                for v in ver:
                    for t in trans:
                        if h :
                            img = img[:, ::-1, :]
                        if v:
                            img = img[::-1, :, :]
                        if t:
                            img = img.transpose(1,0,2)

                        img_tensor = paddle.to_tensor(img)
                        img_tensor /= 255.0
                        
                        img_tensor = paddle.transpose(img_tensor, [2, 0, 1])
                        model.eval()
                        img_tensor = img_tensor.unsqueeze(0)
                        img_out = model(img_tensor)
                        img_out = img_out.squeeze(0).numpy()
                        
                        img_out = img_out.transpose(1, 2, 0)
                        
                        #输出的翻转顺序跟输入相反
                        if t:
                            img_out = img_out.transpose(1,0,2)
                            img = img.transpose(1,0,2)
                        if v:
                            img_out = img_out[::-1, :, :]
                            img = img[::-1, :, :]
                        if h:
                            img_out = img_out[:, ::-1, :]
                            img = img[:, ::-1, :]

                        #最后再slice
                        
                        img_out = img_out[:height, :width, :].clip(0,1)
                        result += img_out

            result /= 8
            
            
            result = result * 255.0
            result = result.round().clip(0,255).astype(np.uint8)
            end_time = time.time()

            s_t = end_time - start_time
            print('time: ' , s_t)
            total_time += s_t
            save_path = "output/new_testB"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_path, im.split('/')[-1]), result)
        print('avg time:' , total_time /200)


if __name__ == '__main__':
    args = parse_args()
    main(args)

