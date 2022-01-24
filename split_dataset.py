import os
import random


random.seed(2021)

image_dir = "datasets/LCDmoire_val/images"
image_paths = os.listdir(image_dir)

random.shuffle(image_paths)
# print(image_paths)

length = len(image_paths)

# train_path = open('train_95.txt', 'w')
val_path = open('lcd_val.txt', 'w')


for i in range(length):
    # if i < int(0.95*length):
    #     print(os.path.join(image_dir, image_paths[i]))  
    #     train_path.write(os.path.join(image_dir, image_paths[i]))
    #     train_path.write('\n')
    # else:
    val_path.write(os.path.join(image_dir, image_paths[i]))
    val_path.write('\n')
