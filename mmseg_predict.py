import os
import numpy as np
import cv2
from tqdm import tqdm

from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv

import matplotlib.pyplot as plt

# 模型 config 配置文件
config_file = 'Zihao-Configs/ZihaoDataset_KNet_20230818.py'

# 模型 checkpoint 权重文件
checkpoint_file = 'checkpoint/Zihao_KNet.pth'

# 计算硬件
# device = 'cpu'
device = 'cuda:0'
model = init_model(config_file, checkpoint_file, device=device)

# 每个类别的 BGR 配色
palette = [
    ['background', [0,0,0]],
    ['veg_bed', [0,127,20]],
    ['ditch', [127,0,0]],
    ['cem_grd', [127,127,0]]
]

palette_dict = {}
for idx, each in enumerate(palette):
    palette_dict[idx] = each[1]

#输出分割结果
if not os.path.exists('testset-pred/mask'):
    os.mkdir('testset-pred/mask')
#测试集路径
PATH_IMAGE = 'Watermelon87_Semantic_Seg_Mask/img_dir/val'
os.chdir(PATH_IMAGE)

opacity = 0.3 # 透明度，越大越接近原图

#单张图像
def process_single_img(img_path, save=False):
    img_bgr = cv2.imread(img_path)

    # 语义分割预测
    result = inference_model(model, img_bgr)
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

    # 将预测的整数ID，映射为对应类别的颜色
    pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
    for idx in palette_dict.keys():
        pred_mask_bgr[np.where(pred_mask == idx)] = palette_dict[idx]
    pred_mask_bgr = pred_mask_bgr.astype('uint8')

    # 将语义分割预测图和原图叠加显示
    pred_viz = cv2.addWeighted(img_bgr, opacity, pred_mask_bgr, 1 - opacity, 0)

    # 保存图像至 outputs/testset-pred 目录
    if save:
        save_path = os.path.join('testset-pred', 'mask', img_path.split('/')[-1])
        cv2.imwrite(save_path, pred_viz)

for each in tqdm(os.listdir()):
    process_single_img(each, save=True)

