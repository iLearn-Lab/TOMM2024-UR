##.p files to .mat files
##step1和step2可以分别进行，两者注释掉一个就行
import numpy as np
import torch
from torch.autograd import Variable
from scipy import io
import os
import os.path as osp
import json

import cv2
from tqdm import tqdm


data_json = osp.join('D:/research/data/refcocog/data.json')
print('Loader loading data.json: ', data_json)
info = json.load(open(data_json))

images = info['images']
anns = info['anns']
Images = {image['image_id']: image for image in images}
Anns = {ann['ann_id']: ann for ann in anns}
cat_to_ix = info['cat_to_ix']
ix_to_cat = {ix: cat for cat, ix in cat_to_ix.items()}

#image_id = 571648
#image_id = 573704
image_id = 538872
#image_id = 397390
#image_id = 580277
#image_id = 314247
#image_id = 203036
image = Images[image_id]
ann_ids = image['ann_ids']
image_path = os.path.join('D:/research/data/train2014/train2014', image['file_name'])
img = cv2.imread(image_path)

print(len(ann_ids))
for i, ann_id in enumerate(ann_ids):
    if i == 2:
        ann = Anns[ann_id]
        cat_id = ann['category_id']
        print(i, end=' ')
        print(ix_to_cat[cat_id])
        x1, y1, w, h = ann["box"][0], ann["box"][1],  ann["box"][2], ann["box"][3]
        x1 = int(x1)
        y1 = int(y1)
        w = int(w)
        h = int(h)
        x2 = x1 + w
        y2 = y1 + h
        centerx = int(x1 + 0.5*w)
        centery = int(y1 + 0.5*h)
        # print(img.shape)  # 图片大小

        #font = cv2.FONT_HERSHEY_SIMPLEX
        font = cv2.FONT_HERSHEY_DUPLEX
        #cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        #cv2.putText(img, str(ix_to_cat[cat_id]), (x1-20,y1-10), font, 1.2, (0, 255, 255), 1, cv2.LINE_AA)

        #cv2.circle(img, (centerx,centery), 3, (0,0,255), 3)
        # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        # cv2.putText(img, '(x,y)', (centerx-40,centery-20), font, 1.5, (0, 0,255), 2, cv2.LINE_AA)
        # cv2.putText(img, 'w', (centerx,y1-10), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(img, 'h', (x1-30,centery), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

    if i == 0:
        ann = Anns[ann_id]
        cat_id = ann['category_id']
        print(i, end=' ')
        print(ix_to_cat[cat_id])
        x1, y1, w, h = ann["box"][0], ann["box"][1], ann["box"][2], ann["box"][3]
        x1 = int(x1)
        y1 = int(y1)
        w = int(w)
        h = int(h)
        x2 = x1 + w
        y2 = y1 + h
        centerx = int(x1 + 0.5 * w)
        centery = int(y1 + 0.5 * h)
        # print(img.shape)  # 图片大小

        # font = cv2.FONT_HERSHEY_SIMPLEX
        font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.putText(img, str(ix_to_cat[cat_id]), (x1-10, y1-10), font, 1.2, (0, 255, 255), 1, cv2.LINE_AA)

        # cv2.circle(img, (centerx,centery), 3, (0,0,255), 3)
        # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        # cv2.putText(img, '(x,y)', (centerx-40,centery-20), font, 1.5, (0, 0,255), 2, cv2.LINE_AA)
        # cv2.putText(img, 'w', (centerx,y1-10), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(img, 'h', (x1-30,centery), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow("Demo1", img)
cv2.waitKey(0)
cv2.imwrite('./man.png', img)
print("finish")