# encoding:utf-8

"""
data_json has
0. refs:       [{ref_id, ann_id, box, image_id, split, category_id, sent_ids, att_wds}]
1. images:     [{image_id, ref_ids, file_name, width, height, h5_id}]
2. anns:       [{ann_id, category_id, image_id, box, h5_id}]
3. sentences:  [{sent_id, tokens, h5_id}]
4. word_to_ix: {word: ix}
5. att_to_ix : {att_wd: ix}
6. att_to_cnt: {att_wd: cnt}
7. label_length: L

Note, box in [xywh] format
label_h5 has
/labels is (M, max_length) uint32 array of encoded labels, zeros padded
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import io
import json
#from loaders.loader import Loader
import os
import argparse
import os.path as osp
import numpy as np
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import numpy as np
import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from interpreter import *
from executor import *
from methods import *
from tqdm import tqdm
import time

METHODS_MAP = {
    "baseline": Baseline,
    "random": Random,
    "parse": Parse,
}

# get ann category
def get_ann_boxes(image_id, Images, Anns):
    image = Images[image_id]
    ann_ids = image['ann_ids']
    ann_boxes = []

    for ann_id in ann_ids:
        ann = Anns[ann_id]
        gold_box = Box(x=ann["box"][0], y=ann["box"][1], w=ann["box"][2], h=ann["box"][3])
        ann_boxes.append( gold_box )
    return ann_boxes                     #该图片所有ann_box类别单词id 有的类别可能含有多个单词

# get category for each sentence

def main(args):
    # dataset_splitBy
    # max_length

    # load data
    #data_json = osp.join('/home/share2/zpp/MAttNet/cache/prepro', dataset_splitBy, 'data.json')
    data_json = osp.join('refcocog/data.json')
    print('Loader loading data.json: ', data_json)
    info = json.load(open(data_json))
    cat_to_ix = info['cat_to_ix']
    ix_to_cat = {ix: cat for cat, ix in cat_to_ix.items()}
    print('object cateogry size is ', len(ix_to_cat))
    images = info['images']
    anns = info['anns']
    refs = info['refs']
    sentences = info['sentences']
    print('we have %s images.' % len(images))
    print('we have %s anns.' % len(anns))
    print('we have %s refs.' % len(refs))
    print('we have %s sentences.' % len(sentences))

    # construct mapping
    method = METHODS_MAP[args.method](args)
    #text_list = json.load(open('/home/share2/zpp/CLIP/vocab_NN_glove.json'))
    text_list = json.load(open('refcocog/refcocog_vocab_NN_glove.json'))
    text_list = [noun.lower() for noun in text_list]

    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    print(device)
    executor = ClipExecutor(clip_model=args.clip_model, box_representation_method=args.box_representation_method,
                            method_aggregator=args.box_method_aggregator, device=device,
                            square_size=not args.non_square_size,
                            expand_position_embedding=args.expand_position_embedding, blur_std_dev=args.blur_std_dev,
                            cache_path=args.cache_path)


    Images = {image['image_id']: image for image in images}
    Anns = {ann['ann_id']: ann for ann in anns}
    #image_root = '/home/share2/zpp/DTWREG_cp/pyutils/refer/data/images/mscoco/images/train2014'
    #image_root = '/home/share/zhangpanpan/Mattnet/MAttNet/data/images/mscoco/images/train2014'
    image_root = '/home/xsq/zpp/train2014'

    imgs_noun = {}
    imgs_noun_sort_val = {}
    imgs_noun_sort_idx = {}
    tic = time.time()

    for image in tqdm(images[17199:]):                    #遍历图片
        #print(time.time() - tic)
        tic = time.time()
        image_id = image['image_id']        #取一张图片
        img_path = os.path.join(image_root, image['file_name'])
        img = Image.open(img_path).convert('RGB')

        # get ann category
        boxes = get_ann_boxes(image_id, Images, Anns)
        env = Environment(img, boxes, executor, (args.mdetr is not None and not args.mdetr_given_bboxes),
                          str(image_id))

        result = method.execute(text_list, env)   #(ann, noun)
        values, idxs = torch.sort(result, dim=-1, descending=True)


        result = result.cpu().numpy().tolist()
        values = values.cpu().numpy().tolist()
        idxs = idxs.cpu().numpy().tolist()

        imgs_noun[image_id] = result
        imgs_noun_sort_val[image_id] = values
        imgs_noun_sort_idx[image_id] = idxs

    json.dump({'imgs_noun_sort_idx': imgs_noun_sort_idx},
              open(osp.join('refcocog/refcocog_imgs_noun_sort_idx3.json'), 'w'))
    json.dump({'imgs_noun_sort_val': imgs_noun_sort_val},
              open(osp.join('refcocog/refcocog_imgs_noun_sort_val3.json'), 'w'))
    #json.dump({'imgs_noun': imgs_noun},
    #          open(osp.join('refcoco+/refcoco+_imgs_noun2.json'), 'w'))

    print('similarity written!')






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        help="input file with expressions and annotations in jsonlines format")
    parser.add_argument("--image_root", type=str, help="path to images (train2014 directory of COCO)")
    parser.add_argument("--clip_model", type=str, default="RN50x16,ViT-B/32",
                        help="which clip model to use (should use RN50x4, ViT-B/32, or both separated by a comma")
    parser.add_argument("--albef_path", type=str, default=None,
                        help="to use ALBEF (instead of CLIP), specify the path to the ALBEF checkpoint")
    parser.add_argument("--method", type=str, default="parse", help="method to solve expressions")
    parser.add_argument("--box_representation_method", type=str, default="crop,blur",
                        help="method of representing boxes as individual images (crop, blur, or both separated by a comma)")
    parser.add_argument("--box_method_aggregator", type=str, default="sum",
                        help="method of combining box representation scores")
    parser.add_argument("--box_area_threshold", type=float, default=0.0,
                        help="minimum area (as a proportion of image area) for a box to be considered as the answer")
    parser.add_argument("--output_file", type=str, default=None, help="(optional) output path to save results")
    parser.add_argument("--detector_file", type=str, default=None,
                        help="(optional) file containing object detections. if not provided, the gold object boxes will be used.")
    parser.add_argument("--mock", action="store_true", help="(optional) mock CLIP execution.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device to use.")
    parser.add_argument("--shuffle_words", action="store_true", help="If true, shuffle words in the sentence")
    parser.add_argument("--gradcam_alpha", type=float, nargs='+', help="alpha value to use for gradcam method")
    parser.add_argument("--enlarge_boxes", type=float, default=0.0,
                        help="(optional) whether to enlarge boxes when passing them to the model")
    parser.add_argument("--part", type=str, default=None,
                        help="(optional) specify how many parts to divide the dataset into and which part to run in the format NUM_PARTS,PART_NUM")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="number of instances to process in one model call (only supported for baseline model)")
    parser.add_argument("--baseline_head", action="store_true",
                        help="For baseline, controls whether model is called on both full expression and head noun chunk of expression")
    parser.add_argument("--mdetr", type=str, default=None,
                        help="to use MDETR as the executor model, specify the name of the MDETR model")
    parser.add_argument("--albef_block_num", type=int, default=8, help="block num for ALBEF gradcam")
    parser.add_argument("--albef_mode", type=str, choices=["itm", "itc"], default="itm")
    parser.add_argument("--expand_position_embedding", action="store_true")
    parser.add_argument("--gradcam_background", action="store_true")
    parser.add_argument("--mdetr_given_bboxes", action="store_true")
    parser.add_argument("--mdetr_use_token_mapping", action="store_true")
    parser.add_argument("--non_square_size", action="store_true")
    parser.add_argument("--blur_std_dev", type=int, default=100, help="standard deviation of Gaussian blur")
    parser.add_argument("--gradcam_ensemble_before", action="store_true",
                        help="Average gradcam maps of different models before summing over the maps")
    parser.add_argument("--cache_path", type=str, default=None, help="cache features")
    # Arguments related to Parse method.
    parser.add_argument("--no_rel", action="store_true", help="Disable relation extraction.")
    parser.add_argument("--no_sup", action="store_true", help="Disable superlative extraction.")
    parser.add_argument("--no_null", action="store_true", help="Disable null keyword heuristics.")
    parser.add_argument("--ternary", action="store_true", help="Disable ternary relation extraction.")
    parser.add_argument("--baseline_threshold", type=float, default=float("inf"),
                        help="(Parse) Threshold to use relations/superlatives.")
    parser.add_argument("--temperature", type=float, default=1., help="(Parse) Sigmoid temperature.")
    parser.add_argument("--superlative_head_only", action="store_true",
                        help="(Parse) Superlatives only quanntify head predicate.")
    parser.add_argument("--sigmoid", action="store_true", help="(Parse) Use sigmoid, not softmax.")
    parser.add_argument("--no_possessive", action="store_true",
                        help="(Parse) Model extraneous relations as possessive relations.")
    parser.add_argument("--expand_chunks", action="store_true",
                        help="(Parse) Expand noun chunks to include descendant tokens that aren't ancestors of tokens in other chunks")
    parser.add_argument("--parse_no_branch", action="store_true",
                        help="(Parse) Only do the parsing procedure if some relation/superlative keyword is in the expression")
    parser.add_argument("--possessive_no_expand", action="store_true", help="(Parse) Expand ent2 in possessive case")

    parser.add_argument('--data_root', default='data', type=str,
                        help='data folder containing images and four datasets.')
    parser.add_argument('--dataset', default='refcoco', type=str, help='refcoco/refcoco+/refcocog')
    parser.add_argument('--splitBy', default='unc', type=str, help='unc/google')

    # argparse
    args = parser.parse_args()
    # call main
    main(args)