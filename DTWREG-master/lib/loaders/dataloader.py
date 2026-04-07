# encoding: utf-8
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

import os.path as osp
import numpy as np
import h5py
import json
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
from loaders.loader import Loader
from mrcn import inference_no_imdb
import functools
import pdb


# box functions
def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))


def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))


def load_vocab_dict_from_file(dict_file):
    if (sys.version_info > (3, 0)):
        with open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
    else:
        with io.open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
    vocab_dict = {n: words[n] for n in range(len(words))}
    return vocab_dict


def load_vocab_dict_from_file2(dict_file):
    if (sys.version_info > (3, 0)):
        with open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
    else:
        with io.open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
    vocab_dict = {words[n]: n for n in range(len(words))}
    return vocab_dict


class DataLoader(Loader):

    def __init__(self, data_json, data_h5, sub_obj_wds, similarity, opt, sent_extract, label_type):
        # parent loader instance
        Loader.__init__(self, data_json, sub_obj_wds, similarity, data_h5)
        # prepare attributes
        self.att_to_ix = self.info['att_to_ix']
        self.ix_to_att = {ix: wd for wd, ix in self.att_to_ix.items()}
        self.num_atts = len(self.att_to_ix)
        self.att_to_cnt = self.info['att_to_cnt']

        self.sent_extract = sent_extract
        self.embedmat_path = '/home/share2/zpp/REC/DTWREG_CLIP/cache/word_embedding/embed_matrix.npy'
        self.embedding_mat = np.load(self.embedmat_path)

        #img_anns_nouns_glove_ids
        self.vocab_file = '/home/share2/zpp/REC/DTWREG_CLIP/cache/word_embedding/vocabulary_72700.txt'
        self.vocab_dict = load_vocab_dict_from_file(self.vocab_file)
        self.vocabTo_ix = load_vocab_dict_from_file2(self.vocab_file)


        #category
        # self.filename = opt['dataset'] + '_imgs_anns_top_train_nouns_glove_ids.json'
        # self.keys = opt['dataset'] + '_imgs_anns_top_train_nouns_glove_ids'
        # self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
        #                                  self.filename)
        # self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
        # self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))

        # self.filename = opt['dataset'] + '_crop_imgs_anns_top_train_nouns_glove_ids.json'
        # self.keys = 'imgs_anns_top_train_nouns_glove_ids'
        # self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/crop',
        #                                  self.filename)
        # self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
        # self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))

        #
        # self.filename = opt['dataset'] + '_Vi_imgs_anns_top_train_nouns_glove_ids.json'
        # self.keys = 'imgs_anns_top_train_nouns_glove_ids'
        # self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache', opt['dataset'],
        #                                  self.filename)
        # self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
        # self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))

        if label_type == 'crop':
            self.filename = opt['dataset'] + '_crop_imgs_anns_top_train_nouns_glove_ids.json'
            self.keys = 'imgs_anns_top_train_nouns_glove_ids'
            self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/crop',
                                             self.filename)
            self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
            self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))

        elif label_type == 'blur':
            self.filename = opt['dataset'] + '_blur_imgs_anns_top_train_nouns_glove_ids.json'
            self.keys = 'imgs_anns_top_train_nouns_glove_ids'
            self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/blur',
                                             self.filename)
            self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
            self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))

        elif label_type == 'RN':
            self.filename = opt['dataset'] + '_RN_imgs_anns_top_train_nouns_glove_ids.json'
            self.keys = 'imgs_anns_top_train_nouns_glove_ids'
            self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache', opt['dataset'],
                                             self.filename)
            self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
            self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))

        elif label_type == 'Vi':
            self.filename = opt['dataset'] + '_Vi_imgs_anns_top_train_nouns_glove_ids.json'
            self.keys = 'imgs_anns_top_train_nouns_glove_ids'
            self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache', opt['dataset'],
                                             self.filename)
            self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
            self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))

        elif label_type == 'Vi_blur':
            self.filename = opt['dataset'] + '_Vi_blur_imgs_anns_top_train_nouns_glove_ids.json'
            self.keys = 'imgs_anns_top_train_nouns_glove_ids'
            self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
                                             self.filename)
            self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
            self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))

        elif label_type == 'Vi_crop':
            self.filename = opt['dataset'] + '_Vi_crop_imgs_anns_top_train_nouns_glove_ids.json'
            self.keys = 'imgs_anns_top_train_nouns_glove_ids'
            self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
                                             self.filename)
            self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
            self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))

        elif label_type == 'RN_crop':
            self.filename = opt['dataset'] + '_RN_crop_imgs_anns_top_train_nouns_glove_ids.json'
            self.keys =  'imgs_anns_top_train_nouns_glove_ids'
            self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
                                             self.filename)
            self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
            self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))

        elif label_type == 'RN_blur':
            self.filename = opt['dataset'] + '_RN_blur_imgs_anns_top_train_nouns_glove_ids.json'
            self.keys = 'imgs_anns_top_train_nouns_glove_ids'
            self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
                                             self.filename)
            self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
            self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))


        elif label_type == 'raw':
            self.filename = opt['dataset'] + '_imgs_anns_top_train_nouns_glove_ids.json'
            self.keys = opt['dataset'] + '_imgs_anns_top_train_nouns_glove_ids'
            self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
                                             self.filename)
            self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
            self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))

        print(self.filename)




        #color img_anns_colors_glove_ids
        # self.filename2 = opt['dataset'] + '_anns_color_glove_id.json'
        # self.keys2 = 'img_ann_glove_id'
        # self.anns_color_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
        #                                     self.filename2)
        # self.anns_color_glove_id = json.load(open(self.anns_color_glove_id))
        # self.anns_color_glove_id = self.anns_color_glove_id[self.keys2]  # (img_id:(ann,txt))

        self.filename2 = opt['dataset'] + '_new_anns_colors_glove_ids.json'
        self.keys2 = 'imgs_anns_colors'
        self.anns_color_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
                                            self.filename2)
        self.anns_color_glove_id = json.load(open(self.anns_color_glove_id))
        self.anns_color_glove_id = self.anns_color_glove_id[self.keys2]  # (img_id:(ann,txt))



        # sent_color_id
        self.sents_att = json.load(open(osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/sent_color_wordid',
                                                 opt['dataset'] + '_' + opt['splitBy'],
                                                 opt['dataset']+ '_sent_to_color_id.json' ) ) )

        # sent_To_glove_ids
        self.filename4 = opt['dataset'] + '_sentToglove_ids.json'
        self.keys4 = 'sentToglove_ids'
        self.sentToglove_ids = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
                                            self.filename4)
        self.sentToglove_ids = json.load(open(self.sentToglove_ids))
        self.sentToglove_ids = self.sentToglove_ids[self.keys4]  # (img_id:(ann,txt))




        # self.filename = opt['dataset'] + '_blur_imgs_anns_top_train_nouns_glove_ids.json'
        # self.keys = 'imgs_anns_top_train_nouns_glove_ids'
        # self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/blur',
        #                                  self.filename)
        # self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
        # self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))

        # self.filename = opt['dataset'] + '_imgs_anns_top_train_nouns_glove_ids.json'
        # self.keys = opt['dataset'] + '_imgs_anns_top_train_nouns_glove_ids'
        # self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
        #                                  self.filename)
        # self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
        # self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))

        # self.filename = opt['dataset'] + '_imgs_anns_train_r1r6_glove_ids.json'
        # self.keys = 'img_ann_glove_id'
        # self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
        #                                  self.filename)
        # self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
        # self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))

        # self.filename = opt['dataset'] + '_imgs_anns_train_r1r6_glove_ids.json'
        # self.keys = 'img_ann_glove_id'
        # self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
        #                                  self.filename)
        # self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
        # self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))

        # imge_anns_colors_vals
        # self.filename3 = opt['dataset'] + '_anns_color_sort_val.json'
        # self.keys3 = 'imgs_noun_sort_val'
        # self.anns_color_sort_val = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
        #                                     self.filename3)
        # self.anns_color_sort_val = json.load(open(self.anns_color_sort_val))
        # self.anns_color_sort_val = self.anns_color_sort_val[self.keys3]  # (img_id:(ann,txt))

        # old
        # self.sents_att = json.load(open(osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/sent_color_wordid',
        #                                          opt['dataset'] + '_' + opt['splitBy'],
        #                                          'sent_color_wordid')))

        # self.filename = opt['dataset'] + '_anns_nouns_glove_ids15.json'
        # self.keys = 'imgs_anns_top_train_nouns_glove_ids'
        # self.img_ann_glove_id = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
        #                                  self.filename)
        # self.img_ann_glove_id = json.load(open(self.img_ann_glove_id))
        # self.img_ann_glove_id = self.img_ann_glove_id[self.keys]  # (img_id:(ann,txt))

        # img_iterators for each split
        self.split_ix = {}
        self.iterators = {}
        for image_id, image in self.Images.items():
            # we use its ref's split (there is assumption that each image only has one split)
            split = self.Refs[image['ref_ids'][0]]['split']
            if split not in self.split_ix:
                self.split_ix[split] = []
                self.iterators[split] = 0
            self.split_ix[split] += [image_id]
        for k, v in self.split_ix.items():
            print('assigned %d images to split %s' % (len(v), k))

    def prepare_mrcn(self, head_feats_dir, args):
        """
        Arguments:
          head_feats_dir: cache/feats/dataset_splitBy/net_imdb_tag, containing all image conv_net feats
          args: imdb_name, net_name, iters, tag
        """
        self.head_feats_dir = head_feats_dir
        self.mrcn = inference_no_imdb.Inference(args)
        assert args.net_name == 'res101'
        self.pool5_dim = 1024
        self.fc7_dim = 2048

    # load different kinds of feats
    def loadFeats(self, Feats):
        # Feats = {feats_name: feats_path}
        self.feats = {}
        self.feat_dim = None
        for feats_name, feats_path in Feats.items():
            if osp.isfile(feats_path):
                self.feats[feats_name] = h5py.File(feats_path, 'r')
                self.feat_dim = self.feats[feats_name]['fc7'].shape[1]
                assert self.feat_dim == self.fc7_dim
                print('FeatLoader loading [%s] from %s [feat_dim %s]' % (feats_name, feats_path, self.feat_dim))

    # shuffle split
    def shuffle(self, split):
        random.shuffle(self.split_ix[split])

    # reset iterator
    def resetIterator(self, split):
        self.iterators[split] = 0

    # expand list by seq per ref, i.e., [a,b], 3 -> [aaabbb]
    def expand_list(self, L, n):
        out = []
        for l in L:
            out += [l] * n
        return out

    def image_to_head(self, image_id):
        """Returns
        head: float32 (1, 1024, H, W)
        im_info: float32 [[im_h, im_w, im_scale]]
        """
        feats_h5 = osp.join(self.head_feats_dir, str(image_id) + '.h5')
        feats = h5py.File(feats_h5, 'r')
        head, im_info = feats['head'], feats['im_info']
        return np.array(head), np.array(im_info)

    def fetch_neighbour_ids(self, ann_id):
        """
        For a given ann_id, we return
        - st_ann_ids: same-type neighbouring ann_ids (not include itself)
        - dt_ann_ids: different-type neighbouring ann_ids
        Ordered by distance to the input ann_id
        """
        ann = self.Anns[ann_id]
        x, y, w, h = ann['box']
        rx, ry = x + w / 2, y + h / 2

        @functools.cmp_to_key
        def compare(ann_id0, ann_id1):
            x, y, w, h = self.Anns[ann_id0]['box']
            ax0, ay0 = x + w / 2, y + h / 2
            x, y, w, h = self.Anns[ann_id1]['box']
            ax1, ay1 = x + w / 2, y + h / 2
            # closer to farmer
            if (rx - ax0) ** 2 + (ry - ay0) ** 2 <= (rx - ax1) ** 2 + (ry - ay1) ** 2:
                return -1
            else:
                return 1

        image = self.Images[ann['image_id']]

        ann_ids = list(image['ann_ids'])
        ann_ids = sorted(ann_ids, key=compare)

        st_ann_ids, dt_ann_ids = [], []
        for ann_id_else in ann_ids:
            if ann_id_else != ann_id:
                if self.Anns[ann_id_else]['category_id'] == ann['category_id']:
                    st_ann_ids += [ann_id_else]
                else:
                    dt_ann_ids += [ann_id_else]
        return st_ann_ids, dt_ann_ids

    def fetch_grid_feats(self, boxes, net_conv, im_info):
        """returns -pool5 (n, 1024, 7, 7) -fc7 (n, 2048, 7, 7)"""
        pool5, fc7 = self.mrcn.box_to_spatial_fc7(net_conv, im_info, boxes)
        return pool5, fc7

    def compute_lfeats(self, ann_ids):
        # return ndarray float32 (#ann_ids, 5)
        lfeats = np.empty((len(ann_ids), 5), dtype=np.float32)
        box_center_lfeats = np.zeros((len(ann_ids), 2), dtype=np.float32)
        box_area = np.zeros((len(ann_ids)), dtype=np.float32)
        for ix, ann_id in enumerate(ann_ids):
            ann = self.Anns[ann_id]
            image = self.Images[ann['image_id']]
            x, y, w, h = ann['box']
            ih, iw = image['height'], image['width']
            lfeats[ix] = np.array([x / iw, y / ih, (x + w - 1) / iw, (y + h - 1) / ih, w * h / (iw * ih)], np.float32)
            box_center_lfeats[ix, :] = (x + w / 2) / iw, (y + h / 2) / ih
            box_area[ix] = (w * h) / (ih * iw)
        return lfeats, box_center_lfeats, box_area

    def compute_dif_lfeats(self, ann_ids, topK=5):
        # return ndarray float32 (#ann_ids, 5*topK)
        dif_lfeats = np.zeros((len(ann_ids), 5 * topK), dtype=np.float32)
        for i, ann_id in enumerate(ann_ids):
            # reference box
            rbox = self.Anns[ann_id]['box']
            rcx, rcy, rw, rh = rbox[0] + rbox[2] / 2, rbox[1] + rbox[3] / 2, rbox[2], rbox[3]
            st_ann_ids, _ = self.fetch_neighbour_ids(ann_id)
            # candidate box
            for j, cand_ann_id in enumerate(st_ann_ids[:topK]):
                cbox = self.Anns[cand_ann_id]['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                dif_lfeats[i, j * 5:(j + 1) * 5] = np.array(
                    [(cx1 - rcx) / rw, (cy1 - rcy) / rh, (cx1 + cw - rcx) / rw, (cy1 + ch - rcy) / rh,
                     cw * ch / (rw * rh)])
        return dif_lfeats

    def fetch_attribute_label(self, ref_ids):
        """Return
    - labels    : Variable float (N, num_atts)
    - select_ixs: Variable long (n, )
    """
        labels = np.zeros((len(ref_ids), self.num_atts))
        select_ixs = []
        for i, ref_id in enumerate(ref_ids):
            ref = self.Refs[ref_id]
            if len(ref['att_wds']) > 0:
                select_ixs += [i]
                for wd in ref['att_wds']:
                    labels[i, self.att_to_ix[wd]] = 1

        return Variable(torch.from_numpy(labels).float().cuda()), Variable(torch.LongTensor(select_ixs).cuda())

    def fetch_cxt_feats(self, ann_ids, opt):
        """
        Return
        - cxt_feats : ndarray (#ann_ids, fc7_dim)
        - cxt_lfeats: ndarray (#ann_ids, ann_ids, 5)
        - dist: ndarray (#ann_ids, ann_ids, 1)
        Note we only use neighbouring "different" (+ "same") objects for computing context objects, zeros padded.
        """

        cxt_feats = np.zeros((len(ann_ids), self.fc7_dim), dtype=np.float32)
        cxt_lfeats = np.zeros((len(ann_ids), len(ann_ids), 5), dtype=np.float32)
        dist = np.zeros((len(ann_ids), len(ann_ids), 1), dtype=np.float32)

        for i, ann_id in enumerate(ann_ids):
            # reference box
            rbox = self.Anns[ann_id]['box']
            rcx, rcy, rw, rh = rbox[0] + rbox[2] / 2, rbox[1] + rbox[3] / 2, rbox[2], rbox[3]
            for j, cand_ann_id in enumerate(ann_ids):
                cand_ann = self.Anns[cand_ann_id]
                # fc7_feats
                cxt_feats[i, :] = self.feats['ann']['fc7'][cand_ann['h5_id'], :]
                cbox = cand_ann['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                ccx, ccy = cbox[0] + cbox[2] / 2, cbox[1] + cbox[3] / 2
                dist[i, j, :] = np.array(abs(rcx - ccx) + abs(rcy - ccy))
                cxt_lfeats[i, j, :] = np.array(
                    [(cx1 - rcx) / rw, (cy1 - rcy) / rh, (cx1 + cw - rcx) / rw, (cy1 + ch - rcy) / rh,
                     cw * ch / (rw * rh)])
        return cxt_feats, cxt_lfeats, dist

    def extract_ann_features(self, image_id, opt):
        """Get features for all ann_ids in an image"""
        image = self.Images[image_id]
        ann_ids = image['ann_ids']

        # fetch image features
        head, im_info = self.image_to_head(image_id)
        head = Variable(torch.from_numpy(head).cuda())

        # fetch ann features
        ann_boxes = xywh_to_xyxy(np.vstack([self.Anns[ann_id]['box'] for ann_id in ann_ids]))
        ann_pool5, ann_fc7 = self.fetch_grid_feats(ann_boxes, head, im_info)

        # absolute location features
        lfeats, box_center_lfeats, box_area = self.compute_lfeats(ann_ids)
        lfeats = Variable(torch.from_numpy(lfeats).cuda())
        box_center_lfeats = Variable(torch.from_numpy(box_center_lfeats).cuda())
        box_area = Variable(torch.from_numpy(box_area).cuda())

        # relative location features
        dif_lfeats = self.compute_dif_lfeats(ann_ids)
        dif_lfeats = Variable(torch.from_numpy(dif_lfeats).cuda())

        # fetch context_fc7 and context_lfeats
        cxt_fc7, cxt_lfeats, dist = self.fetch_cxt_feats(ann_ids, opt)
        cxt_fc7 = Variable(torch.from_numpy(cxt_fc7).cuda())
        cxt_lfeats = Variable(torch.from_numpy(cxt_lfeats).cuda())
        dist = Variable(torch.from_numpy(dist).cuda())

        return ann_fc7, ann_pool5, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, dist, box_center_lfeats, box_area

    def compute_rel_lfeats(self, sub_ann_id, obj_ann_id):
        if sub_ann_id == -1 or obj_ann_id == -1:
            rel_lfeats = torch.zeros(5)
        else:
            rbox = self.Anns[sub_ann_id]['box']
            rcx, rcy, rw, rh = rbox[0] + rbox[2] / 2, rbox[1] + rbox[3] / 2, rbox[2], rbox[3]
            cbox = self.Anns[obj_ann_id]['box']
            cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
            rel_lfeats = np.array(
                [(cx1 - rcx) / rw, (cy1 - rcy) / rh, (cx1 + cw - rcx) / rw, (cy1 + ch - rcy) / rh, cw * ch / (rw * rh)])
            rel_lfeats = torch.Tensor(rel_lfeats)
        return rel_lfeats

    # get batch of data
    def getBatch(self, split, opt):
        # options
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1
        wrapped = False
        TopK = opt['num_cxt']

        dataset = opt["dataset"]
        splitBy = opt["splitBy"]

        f = open(self.vocab_file, "r")
        vocab_list = f.read().splitlines()
        f.close()

        ############# load new file #########################

        sent_sub_wordid = self.sent_extract["sent_sub_wordid"]
        # sent_sub_nounid = sent_extract["sent_sub_nounid"]
        # sent_sub_classid = sent_extract["sent_sub_classid"]
        sent_sub_classwordid = self.sent_extract["sent_sub_classwordid"]
        sent_obj_wordid = self.sent_extract["sent_obj_wordid"]
        # sent_obj_nounid = sent_extract["sent_obj_nounid"]
        sent_rel_wordid = self.sent_extract["sent_rel_wordid"]
        # sent_rel_prepid = sent_extract["sent_rel_prepid"]

        #####################################################

        ri = self.iterators[split]
        ri_next = ri + 1
        if ri_next > max_index:
            ri_next = 0
            wrapped = True
        self.iterators[split] = ri_next
        image_id = split_ix[ri]

        ann_ids = self.Images[image_id]['ann_ids']
        ann_num = len(ann_ids)
        ref_ids = self.Images[image_id]['ref_ids']

        img_ref_ids = []
        img_sent_ids = []
        gd_ixs = []
        gd_boxes = []
        for ref_id in ref_ids:
            ref = self.Refs[ref_id]
            for sent_id in ref['sent_ids']:
                img_ref_ids += [ref_id]
                img_sent_ids += [sent_id]
                gd_ixs += [ann_ids.index(ref['ann_id'])]
                gd_boxes += [ref['box']]
        img_sent_num = len(img_sent_ids)

        # sub_sim = torch.Tensor(self.similarity[str(image_id)]['sub_sim']).cuda().squeeze(2)
        # obj_sim = torch.Tensor(self.similarity[str(image_id)]['obj_sim']).cuda().squeeze(2)
        # sub_emb = torch.Tensor(self.similarity[str(image_id)]['sub_emb']).cuda()
        # obj_emb = torch.Tensor(self.similarity[str(image_id)]['obj_emb']).cuda()

        ann_fc7, ann_pool5, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, dist, box_center_lfeats, box_area = self.extract_ann_features(
            image_id, opt)

        pool5 = ann_pool5.unsqueeze(0).expand(img_sent_num, ann_num, self.pool5_dim, 7, 7)
        pool5.detach()

        fc7 = ann_fc7.unsqueeze(0).expand(img_sent_num, ann_num, self.fc7_dim, 7, 7)
        fc7.detach()

        ann_fleats = lfeats
        lfeats = lfeats.unsqueeze(0).expand(img_sent_num, ann_num, 5)
        lfeats.detach()

        dif_lfeats = dif_lfeats.unsqueeze(0).expand(img_sent_num, ann_num, TopK * 5)
        dif_lfeats.detach()

        cxt_fc7.detach()
        cxt_lfeats.detach()
        dist.detach()
        box_center_lfeats.detach()
        box_area.detach()
        # sub_emb.detach()
        # obj_emb.detach()
        # sub_sim.detach()
        # obj_sim.detach()

        att_labels, select_ixs = self.fetch_attribute_label(img_ref_ids)

        labels = np.vstack([self.fetch_seq(sent_id) for sent_id in img_sent_ids])

        labels = Variable(torch.from_numpy(labels).long().cuda())
        max_len = (labels != 0).sum(1).max().data[0]
        labels = labels[:, :max_len]

        start_words = np.ones([labels.size(0), 1], dtype=int) * (self.word_to_ix['<BOS>'])
        start_words = Variable(torch.from_numpy(start_words).long().cuda())
        enc_labels = labels.clone()
        enc_labels = torch.cat([start_words, enc_labels], 1)

        zero_pad = np.zeros([labels.size(0), 1], dtype=int)
        zero_pad = Variable(torch.from_numpy(zero_pad).long().cuda())
        dec_labels = labels.clone()
        dec_labels = torch.cat([dec_labels, zero_pad], 1)

        ###### new data #######
        sub_wordids = []
        sub_classwordids = []
        obj_wordids = []
        rel_wordids = []

        # sub_classids = []
        # sub_nounids = []
        # obj_nounids = []
        # rel_prepids = []

        sub_classembs = []
        sub_wordembs = []
        obj_wordembs = []
        rel_wordembs = []

        ########################
        # panpan
        anns_glove_id = self.img_ann_glove_id[str(image_id)]  # list(ann, txt)每张图片的标注框对应的top20文本id
        anns_glove_emb = [self.embedding_mat[ann_glove_id] for ann_glove_id in anns_glove_id]  # (ann_num, txt_num, dim)
        anns_glove_emb = torch.Tensor(np.array(anns_glove_emb)).cuda()
        anns_words = [[self.vocab_dict[wd_id] for wd_id in ann_glove_id] for ann_glove_id in anns_glove_id]

        for sent_id in img_sent_ids:
            # sub_nounid = sent_sub_nounid[unicode(sent_id)]
            # sub_nounids.append(sub_nounid)
            sub_wordid = sent_sub_wordid[str(sent_id)]
            sub_wordids.append(sub_wordid)
            sub_wordemb = self.embedding_mat[sub_wordid]
            sub_wordembs.append(sub_wordemb)

            # sub_classid = sent_sub_classid[unicode(sent_id)]
            # sub_classids.append(sub_classid)
            sub_classwordid = sent_sub_classwordid[str(sent_id)]
            sub_classwordids.append(sub_classwordid)
            sub_classemb = self.embedding_mat[sub_classwordid]
            sub_classembs.append(sub_classemb)

            # obj_nounid = sent_obj_nounid[unicode(sent_id)]
            # obj_nounids.append(obj_nounid)
            obj_wordid = sent_obj_wordid[str(sent_id)]
            obj_wordids.append(obj_wordid)
            obj_wordemb = self.embedding_mat[obj_wordid]
            obj_wordembs.append(obj_wordemb)

            # rel_prep_id = sent_rel_prepid[unicode(sent_id)]
            # rel_prepids.append(rel_prep_id)
            rel_wordid = sent_rel_wordid[str(sent_id)]
            rel_wordids.append(rel_wordid)
            rel_wordemb = self.embedding_mat[rel_wordid]
            rel_wordembs.append(rel_wordemb)

        sub_words = [self.vocab_dict[wd_id] for wd_id in sub_wordids]
        obj_words = [self.vocab_dict[wd_id] for wd_id in obj_wordids]

        data = {}
        data['labels'] = labels
        data['enc_labels'] = enc_labels
        data['dec_labels'] = dec_labels
        data['ref_ids'] = ref_ids
        data['sent_ids'] = img_sent_ids
        data['gd_ixs'] = gd_ixs
        data['gd_boxes'] = gd_boxes
        # data['sim'] = {'sub_sim': sub_sim, 'obj_sim': obj_sim, 'sub_emb': sub_emb, 'obj_emb': obj_emb}
        data['Feats'] = {'fc7': fc7, 'pool5': pool5, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats,
                         'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats, 'dist': dist}
        data['att_labels'] = att_labels
        data['select_ixs'] = select_ixs
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}

        #################### new data ############################

        # data['sub_nounids'] = (torch.Tensor(np.array(sub_nounids)).cuda()).long()
        data['sub_wordids'] = (torch.Tensor(np.array(sub_wordids)).cuda()).long()
        data['sub_wordembs'] = torch.Tensor(np.array(sub_wordembs)).cuda()

        # data['obj_nounids'] = (torch.Tensor(np.array(obj_nounids)).cuda()).long()
        data['obj_wordids'] = (torch.Tensor(np.array(obj_wordids)).cuda()).long()
        data['obj_wordembs'] = torch.Tensor(np.array(obj_wordembs)).cuda()

        # data['rel_prepids'] = (torch.Tensor(np.array(rel_prepids)).cuda()).long()
        data['rel_wordids'] = (torch.Tensor(np.array(rel_wordids)).cuda()).long()
        data['rel_wordembs'] = torch.Tensor(np.array(rel_wordembs)).cuda()

        # data['sub_classids'] = (torch.Tensor(np.array(sub_classids)).cuda()).long()
        data['sub_classwordids'] = (torch.Tensor(np.array(sub_classwordids)).cuda()).long()
        data['sub_classembs'] = torch.Tensor(np.array(sub_classembs)).cuda()

        data['ann_pool5'] = ann_pool5
        data['ann_fc7'] = ann_fc7
        data['ann_fleats'] = ann_fleats

        data['anns_glove_emb'] = anns_glove_emb

        data['anns_words'] = anns_words
        data['sub_words'] = sub_words
        data['obj_words'] = obj_words

        data['img_ref_ids'] = img_ref_ids
        data['img_sent_ids'] = img_sent_ids
        data['image_id'] = image_id
        data['box_center_lfeats'] = box_center_lfeats
        data['box_area'] = box_area

        #########################################################
        return data

    def get_attribute_weights(self, scale=10):
        # weights = \lamda * 1/sqrt(cnt)
        cnts = [self.att_to_cnt[self.ix_to_att[ix]] for ix in range(self.num_atts)]
        cnts = np.array(cnts)
        weights = 1 / cnts ** 0.5
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        weights = weights * (scale - 1) + 1
        return torch.from_numpy(weights).float()

    def decode_attribute_label(self, scores):
        """- scores: Variable (cuda) (n, num_atts) after sigmoid range [0, 1]
           - labels:list of [[att, sc], [att, sc], ...
        """
        scores = scores.data.cpu().numpy()
        N = scores.shape[0]
        labels = []
        for i in range(N):
            label = []
            score = scores[i]
            for j, sc in enumerate(list(score)):
                label += [(self.ix_to_att[j], sc)]
                labels.append(label)
        return labels

    def getTestBatch(self, split, opt):

        ###### new data  ################################
        dataset = opt["dataset"]
        splitBy = opt["splitBy"]

        # embedmat_path = 'cache/word_embedding/embed_matrix.npy'
        # embedding_mat = np.load(embedmat_path)

        # sent_extract = json.load(open(osp.join('cache/sub_obj_wds', dataset + '_' + splitBy, "sent_extract.json")))
        sent_sub_wordid = self.sent_extract["sent_sub_wordid"]
        # sent_sub_nounid = sent_extract["sent_sub_nounid"]
        # sent_sub_classid = sent_extract["sent_sub_classid"]
        sent_sub_classwordid = self.sent_extract["sent_sub_classwordid"]
        sent_obj_wordid = self.sent_extract["sent_obj_wordid"]
        # sent_obj_nounid = sent_extract["sent_obj_nounid"]
        sent_rel_wordid = self.sent_extract["sent_rel_wordid"]
        # sent_rel_prepid = sent_extract["sent_rel_prepid"]
        sent_color_wordid = self.sents_att['sent_to_color_id']

        ###################################################

        TopK = opt['num_cxt']
        wrapped = False
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1
        ri = self.iterators[split]
        ri_next = ri + 1
        if ri_next > max_index:
            ri_next = 0
            wrapped = True
        self.iterators[split] = ri_next
        image_id = split_ix[ri]
        image = self.Images[image_id]
        ann_ids = image['ann_ids']

        sent_ids = []
        gd_ixs = []
        gd_boxes = []
        att_refs = []
        for ref_id in image['ref_ids']:
            ref = self.Refs[ref_id]
            for sent_id in ref['sent_ids']:
                sent_ids += [sent_id]
                gd_ixs += [ann_ids.index(ref['ann_id'])]
                gd_boxes += [ref['box']]
                att_refs += [ref_id]

        expand_ann_ids = []
        for ann_id in ann_ids:
            for i in range(len(ann_ids)):
                expand_ann_ids.append(ann_id)

        # sub_sim = torch.Tensor(self.similarity[str(image_id)]['sub_sim']).cuda().squeeze(2)
        # obj_sim = torch.Tensor(self.similarity[str(image_id)]['obj_sim']).cuda().squeeze(2)
        # sub_emb = torch.Tensor(self.similarity[str(image_id)]['obj_emb']).cuda()
        # obj_emb = torch.Tensor(self.similarity[str(image_id)]['obj_emb']).cuda()

        ann_fc7, ann_pool5, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, dist, box_center_lfeats, box_area = self.extract_ann_features(
            image_id, opt)

        pool5 = ann_pool5.unsqueeze(0)
        pool5.detach()
        fc7 = ann_fc7.unsqueeze(0)
        fc7.detach()

        ann_fleats = lfeats
        lfeats = lfeats.unsqueeze(0)
        lfeats.detach()

        dif_lfeats = dif_lfeats.unsqueeze(0)
        dif_lfeats.detach()
        cxt_fc7.detach()
        cxt_lfeats.detach()
        # sub_emb.detach()
        # obj_emb.detach()
        # sub_sim.detach()
        # obj_sim.detach()
        dist.detach()
        box_center_lfeats.detach()
        box_area.detach()

        labels = np.vstack([self.fetch_seq(sent_id) for sent_id in sent_ids])
        labels = Variable(torch.from_numpy(labels).long().cuda())
        max_len = (labels != 0).sum(1).max().data[0]
        labels = labels[:, :max_len]

        start_words = np.ones([labels.size(0), 1], dtype=int) * (self.word_to_ix['<BOS>'])
        start_words = Variable(torch.from_numpy(start_words).long().cuda())
        enc_labels = labels.clone()
        enc_labels = torch.cat([start_words, enc_labels], 1)

        zero_pad = np.zeros([labels.size(0), 1], dtype=int)
        zero_pad = Variable(torch.from_numpy(zero_pad).long().cuda())
        dec_labels = labels.clone()
        dec_labels = torch.cat([dec_labels, zero_pad], 1)

        att_labels, select_ixs = self.fetch_attribute_label(att_refs)

        ####### new data #########

        # sub_nounids = []
        sub_wordids = []
        sub_wordembs = []

        # sub_classids = []
        sub_classwordids = []
        sub_classembs = []

        # obj_nounids = []
        obj_wordids = []
        obj_wordembs = []

        # rel_prepids = []
        rel_wordids = []
        rel_wordembs = []

        ##########################
        anns_glove_id = self.img_ann_glove_id[str(image_id)]  # list(ann, txt)每张图片的标注框对应的top20文本id
        anns_glove_emb = [self.embedding_mat[ann_glove_id] for ann_glove_id in anns_glove_id]  # (ann_num, txt_num, dim)
        anns_glove_emb = torch.Tensor(np.array(anns_glove_emb)).cuda()
        anns_words = [[self.vocab_dict[wd_id] for wd_id in ann_glove_id] for ann_glove_id in anns_glove_id]

        anns_color_glove_id = self.anns_color_glove_id[str(image_id)]  # list(ann, txt)每张图片的标注框对应的top20文本id
        anns_color_glove_emb = [self.embedding_mat[ann_color_glove_id] for ann_color_glove_id in
                                anns_color_glove_id]  # (ann_num, txt_num, dim)
        anns_color_glove_emb = torch.Tensor(np.array(anns_color_glove_emb)).cuda()

        #anns_color_sort_val = self.anns_color_sort_val[str(image_id)]


        sent_color_ids = []
        sent_color_embs = []

        per_sentToglove_ids = []
        per_sentToglove_embs = []


        for sent_id in sent_ids:
            per_sentToglove_id = self.sentToglove_ids[str(sent_id)]
            per_sentToglove_emb = self.embedding_mat[per_sentToglove_id]   #np.array(word_num, dim)
            per_sentToglove_emb = torch.Tensor( np.array(per_sentToglove_emb) ).cuda()
            per_sentToglove_embs.append(per_sentToglove_emb)   #np.array(word_num, dim)




            color_wordid = sent_color_wordid[str(sent_id)]
            sent_color_ids.append(color_wordid)
            color_emb = self.embedding_mat[color_wordid]
            sent_color_embs.append(color_emb)

            # new part
            # sub_nounid = sent_sub_nounid[unicode(sent_id)]
            # sub_nounids.append(sub_nounid)
            sub_wordid = sent_sub_wordid[str(sent_id)]
            sub_wordids.append(sub_wordid)
            sub_wordemb = self.embedding_mat[sub_wordid]
            sub_wordembs.append(sub_wordemb)

            # sub_classid = sent_sub_classid[unicode(sent_id)]
            # sub_classids.append(sub_classid)
            sub_classwordid = sent_sub_classwordid[str(sent_id)]
            sub_classwordids.append(sub_classwordid)
            sub_classemb = self.embedding_mat[sub_classwordid]
            sub_classembs.append(sub_classemb)

            # obj_nounid = sent_obj_nounid[unicode(sent_id)]
            # obj_nounids.append(obj_nounid)
            obj_wordid = sent_obj_wordid[str(sent_id)]
            obj_wordids.append(obj_wordid)
            obj_wordemb = self.embedding_mat[obj_wordid]
            obj_wordembs.append(obj_wordemb)

            # rel_prepid = sent_rel_prepid[unicode(sent_id)]
            # rel_prepids.append(rel_prepid)
            rel_wordid = sent_rel_wordid[str(sent_id)]
            rel_wordids.append(rel_wordid)
            rel_wordemb = self.embedding_mat[rel_wordid]
            rel_wordembs.append(rel_wordemb)

        sub_words = [self.vocab_dict[wd_id] for wd_id in sub_wordids]
        obj_words = [self.vocab_dict[wd_id] for wd_id in obj_wordids]

        data = {}
        data['image_id'] = image_id
        data['ann_ids'] = ann_ids
        data['sent_ids'] = sent_ids
        data['gd_ixs'] = gd_ixs
        data['gd_boxes'] = gd_boxes
        #data['sim'] = {'sub_sim': sub_sim, 'obj_sim': obj_sim, 'sub_emb': sub_emb, 'obj_emb': obj_emb}
        data['Feats'] = {'fc7': fc7, 'pool5': pool5, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats,
                         'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats, 'dist': dist}
        data['labels'] = labels
        data['enc_labels'] = enc_labels
        data['dec_labels'] = dec_labels
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}
        data['att_labels'] = att_labels
        data['select_ixs'] = select_ixs

        ######################### new data #####################

        # data['sub_nounids'] = (torch.Tensor(np.array(sub_nounids)).cuda()).long()
        data['sub_wordids'] = (torch.Tensor(np.array(sub_wordids)).cuda()).long()
        data['sub_wordembs'] = torch.Tensor(np.array(sub_wordembs)).cuda()

        # data['obj_nounids'] = (torch.Tensor(np.array(obj_nounids)).cuda()).long()
        data['obj_wordids'] = (torch.Tensor(np.array(obj_wordids)).cuda()).long()
        data['obj_wordembs'] = torch.Tensor(np.array(obj_wordembs)).cuda()

        # data['rel_prepids'] = (torch.Tensor(np.array(rel_prepids)).cuda()).long()
        data['rel_wordids'] = (torch.Tensor(np.array(rel_wordids)).cuda()).long()
        data['rel_wordembs'] = torch.Tensor(np.array(rel_wordembs)).cuda()

        # data['sub_classids'] = (torch.Tensor(np.array(sub_classids)).cuda()).long()
        data['sub_classwordids'] = (torch.Tensor(np.array(sub_classwordids)).cuda()).long()
        data['sub_classembs'] = torch.Tensor(np.array(sub_classembs)).cuda()

        data['ann_pool5'] = ann_pool5
        data['ann_fc7'] = ann_fc7
        data['ann_fleats'] = ann_fleats

        data['expand_ann_ids'] = expand_ann_ids

        data['anns_glove_emb'] = anns_glove_emb

        data['anns_words'] = anns_words
        data['sub_words'] = sub_words
        data['obj_words'] = obj_words

        data['img_ref_ids'] = att_refs
        data['img_sent_ids'] = sent_ids
        data['image_id'] = image_id
        data['box_center_lfeats'] = box_center_lfeats
        data['box_area'] = box_area
        data['anns_color_glove_emb'] = anns_color_glove_emb
        data['sent_color_embs'] = torch.Tensor(np.array(sent_color_embs)).cuda()
        data['sent_color_ids'] = (torch.Tensor(np.array(sent_color_ids)).cuda()).long()
        #data['anns_color_sort_val'] = (torch.Tensor(np.array(anns_color_sort_val)).cuda())

        data['per_sentToglove_embs'] = per_sentToglove_embs

        #######################################################

        return data
