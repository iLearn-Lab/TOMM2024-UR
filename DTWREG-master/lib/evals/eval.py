# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import json
import h5py
import time
from pprint import pprint

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pdb


# refcocog


# IoU function
def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0] + box1[2] - 1, box2[0] + box2[2] - 1)
    inter_y2 = min(box1[1] + box1[3] - 1, box2[1] + box2[3] - 1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return float(inter) / union


def eval_split(loader, model, split, opt, pater1, pater2, pater3):
    if not osp.isdir(osp.join('/home/share2/zpp/REC/DTWREG_CLIP/pred_img', opt['dataset'] + '_' + split)):
        os.makedirs(osp.join('/home/share2/zpp/REC/DTWREG_CLIP/pred_img', opt['dataset'] + '_' + split, 'yes1'))
        os.makedirs(osp.join('/home/share2/zpp/REC/DTWREG_CLIP/pred_img', opt['dataset'] + '_' + split, 'yes2'))
        os.makedirs(osp.join('/home/share2/zpp/REC/DTWREG_CLIP/pred_img', opt['dataset'] + '_' + split, 'no'))

    #sent_To_location/size
    # filename5 = opt['dataset'] + '_sent_to_location.json'
    # keys5 = 'sent_to_location'
    # sent_to_location = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
    #                                  filename5)
    # sent_to_location = json.load(open(sent_to_location))
    # sent_to_location = sent_to_location[keys5]  # (img_id:(ann,txt))
    proposal_spatial_scores = {}

    filename5 = opt['dataset'] + '_sent_locations.json'
    sent_to_location = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
                                     filename5)
    sent_to_location = json.load(open(sent_to_location))

    filename6 = opt['dataset'] + '_sent_sizes.json'
    sent_to_size = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/cache/', opt['dataset'],
                                     filename6)
    sent_to_size = json.load(open(sent_to_size))


    #file = '/home/share2/zpp/REC/DTWREG_CLIP/correct_sent_ids_final'+ opt['dataset'] + '_' + split+'.npy'
    # file = '/home/share2/zpp/DTWREG_cp/correct_sent_ids_final'+ opt['dataset'] + '_' + split+'.npy'
    # correct_sents_final = np.load(file).tolist()


    # file ='/home/share2/zpp/reclip_raw/'+ 'reclip__' + opt['dataset'] + '_' + split + '__pred_scores.npy'
    # file = file.lower()
    # reclip_pred_scores = np.load(file)
    # reclip_pred_scores = reclip_pred_scores.item()
    # pdb.set_trace()

    verbose = opt.get('verbose', True)
    num_sents = opt.get('num_sents', -1)
    assert split != 'train', 'Check the evaluation split.'

    model.eval()

    loader.resetIterator(split)
    loss_sum = 0
    loss_evals = 0
    acc = 0
    predictions = []
    finish_flag = False
    model_time = 0
    our_model_time = 0
    DTMR_model_time = 0

    avg_model_time = 0
    avg_our_model_time = 0
    avg_DTMR_model_time = 0

    Sub_anns_sim = {}
    correct_sent_ids = []

    while True:
        data = loader.getTestBatch(split, opt)
        att_weights = loader.get_attribute_weights()
        sent_ids = data['sent_ids']
        Feats = data['Feats']
        labels = data['labels']
        enc_labels = data['enc_labels']
        dec_labels = data['dec_labels']
        image_id = data['image_id']
        ann_ids = data['ann_ids']
        att_labels, select_ixs = data['att_labels'], data['select_ixs']
        # sim = data['sim']

        ######### new data  ############

        # sub_nounids = data['sub_nounids']
        sub_wordids = data['sub_wordids']
        sub_wordembs = data['sub_wordembs']

        # sub_classids = data['sub_classids']
        # sub_classwordids = data['sub_classwordids']
        sub_classembs = data['sub_classembs']

        # obj_nounids = data['obj_nounids']
        # obj_wordids = data['obj_wordids']
        obj_wordembs = data['obj_wordembs']

        # rel_prepids = data['rel_prepids']
        # rel_wordids = data['rel_wordids']
        rel_wordembs = data['rel_wordembs']

        ann_pool5 = data['ann_pool5']
        ann_fc7 = data['ann_fc7']
        ann_fleats = data['ann_fleats']

        expand_ann_ids = data['expand_ann_ids']

        anns_glove_emb = data['anns_glove_emb']

        sub_wordids = data['sub_wordids']
        obj_wordids = data['obj_wordids']
        box_center_lfeats = data['box_center_lfeats']
        box_area = data['box_area']
        anns_color_glove_emb = data['anns_color_glove_emb']
        sent_color_embs = data['sent_color_embs']
        sent_color_ids = data['sent_color_ids']
        #anns_color_sort_val = data['anns_color_sort_val']
        per_sentToglove_embs = data['per_sentToglove_embs']

        ################################
        sentid_to_category_score={}
        for i, sent_id in enumerate(sent_ids):

            ########### new data #################

            # sub_nounid = sub_nounids[i:i+1]
            sub_wordid = sub_wordids[i:i + 1]
            sub_wordemb = sub_wordembs[i:i + 1]

            # sub_classid = sub_classids[i:i + 1]
            # sub_classwordid = sub_classwordids[i:i + 1]
            sub_classemb = sub_classembs[i:i + 1]

            # obj_nounid = obj_nounids[i:i+1]
            obj_wordid = obj_wordids[i:i + 1]
            obj_wordemb = obj_wordembs[i:i + 1]

            # rel_prepid = rel_prepids[i:i+1]
            # rel_wordid = rel_wordids[i:i+1]
            rel_wordemb = rel_wordembs[i:i + 1]

            #######################################

            enc_label = enc_labels[i:i + 1]  # (1, sent_len)
            max_len = (enc_label != 0).sum().data[0]
            enc_label = enc_label[:, :max_len]  # (1, max_len)
            dec_label = dec_labels[i:i + 1]
            dec_label = dec_label[:, :max_len]

            label = labels[i:i + 1]
            max_len = (label != 0).sum().data[0]
            label = label[:, :max_len]  # (1, max_len)

            pool5 = Feats['pool5']
            fc7 = Feats['fc7']
            lfeats = Feats['lfeats']
            dif_lfeats = Feats['dif_lfeats']
            dist = Feats['dist']
            cxt_fc7 = Feats['cxt_fc7']
            cxt_lfeats = Feats['cxt_lfeats']
            # sub_sim = sim['sub_sim'][i:i+1]
            # obj_sim = sim['obj_sim'][i:i+1]
            # # sub_emb = sim['sub_emb'][i:i+1]
            # # obj_emb = sim['obj_emb'][i:i+1]

            att_label = att_labels[i:i + 1]
            if i in select_ixs:
                select_ix = torch.LongTensor([0]).cuda()
            else:
                select_ix = torch.LongTensor().cuda()

            tic = time.time()
            scores, loss, sub_loss, obj_loss, rel_loss, sub_anns_sim,\
            lfeat_score, proposal_spatial_score = model(opt['dataset'], pool5, fc7, lfeats, dif_lfeats,
                                                                             box_center_lfeats, box_area, cxt_fc7,
                                                                             cxt_lfeats, dist, label, enc_label,
                                                                             dec_label, att_label, select_ix,
                                                                             att_weights,
                                                                             sub_wordemb, sub_classemb, obj_wordemb,
                                                                             rel_wordemb,
                                                                             ann_pool5, ann_fc7, ann_fleats, image_id,
                                                                             sent_id, sub_wordid, obj_wordid,
                                                                             anns_glove_emb, anns_color_glove_emb,
                                                                             sent_color_embs[i:i + 1],
                                                                             sent_color_ids[i:i + 1],
                                                                             per_sentToglove_embs[i],
                                                                             sent_to_location[str(sent_id)],
                                                                             sent_to_size[str(sent_id)], pater1, pater2, pater3)
            # reclip_scores = reclip_pred_scores[sent_id].cuda()
            # reclip_scores = reclip_scores.unsqueeze(1).expand(len(ann_ids), len(ann_ids))
            # reclip_scores = reclip_scores.contiguous().view( len(ann_ids)*len(ann_ids), 1).squeeze(1)
            # reclip_scores = reclip_scores.float()

            scores = scores.squeeze(0)
            # pdb.set_trace()
            # scores += 0.5*reclip_scores



            loss = loss.data[0].item()

            pred_ix = torch.argmax(scores)

            pred_ann_id = expand_ann_ids[pred_ix]

            gd_ix = data['gd_ixs'][i]
            loss_sum += loss
            loss_evals += 1

            pred_box = loader.Anns[pred_ann_id]['box']
            gd_box = data['gd_boxes'][i]

            IoU = computeIoU(pred_box, gd_box)

            if opt['use_IoU'] > 0:
                if IoU >= 0.5:
                    acc += 1
            else:
                if pred_ix == gd_ix:
                    acc += 1

            # #可视化操作
            # sent_tokens = loader.Sentences[sent_id]['tokens']
            # save_img_name = str(sent_id) + '_' + ' '.join(sent_tokens) + '.jpg'
            # dets = [pred_box, gd_box]
            # gd_ann_id = ann_ids[gd_ix]
            # pre_cls = loader.ix_to_cat[loader.Anns[pred_ann_id]['category_id']]
            # gd_cls = loader.ix_to_cat[loader.Anns[gd_ann_id]['category_id']]
            # clas = [pre_cls, gd_cls]
            # colors = [(255, 0, 0), (0, 255, 0)]
            # import cv2
            # image_id = loader.Anns[gd_ann_id]['image_id']
            # image_name = '/home/share2/zpp/DTWREG_cp/pyutils/refer/data/images/mscoco/images/train2014/' + \
            #              loader.Images[image_id]['file_name']
            # img = cv2.imread(image_name)
            # for i in range(2):
            #     x1, y1, w, h = dets[i]
            #     cls = clas[i]
            #     x1 = int(x1)
            #     y1 = int(y1)
            #     w = int(w)
            #     h = int(h)
            #     x2 = x1 + w
            #     y2 = y1 + h
            #     # print(img.shape)  # 图片大小
            #     cv2.rectangle(img, (x1, y1), (x2, y2), colors[i], 2)
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
            #     cv2.putText(img, cls, (x1, y1), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            #
            # # cv2.imshow("fff", img)
            # # cv2.waitKey(0)
            #
            # # if pred_ann_id == gd_ann_id:
            # #     correct_sent_ids.append(sent_id)
            # #     save_path = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/pred_img', opt['dataset']+'_'+split, 'yes1', save_img_name)
            # #     cv2.imwrite(save_path, img)
            # # elif pred_ann_id != gd_ann_id and IoU >= 0.5:
            # #     save_path = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/pred_img', opt['dataset']+'_'+split, 'yes2', save_img_name)
            # #     cv2.imwrite(save_path, img)
            # if pred_ann_id != gd_ann_id and IoU < 0.5 and sent_id in correct_sents_final:
            #     save_path = osp.join('/home/share2/zpp/REC/DTWREG_CLIP/pred_img', opt['dataset']+'_'+split, 'no', save_img_name)
            #     cv2.imwrite(save_path, img)

            entry = {}
            entry['image_id'] = image_id
            entry['sent_id'] = sent_id
            entry['sent'] = loader.decode_labels(label.data.cpu().numpy())[0]  # gd-truth sent
            entry['gd_ann_id'] = data['ann_ids'][gd_ix]
            entry['pred_ann_id'] = pred_ann_id
            entry['pred_score'] = scores.tolist()[pred_ix]
            entry['IoU'] = IoU
            entry['ann_ids'] = ann_ids

            predictions.append(entry)
            # model_time += (toc - tic)
            # our_model_time += our_time
            # DTMR_model_time += DTMR_time
            #
            # avg_model_time += (toc - tic)
            # avg_our_model_time += our_time
            # avg_DTMR_model_time += DTMR_time

            proposal_spatial_scores[sent_id] = proposal_spatial_score

            Sub_anns_sim[sent_id] = sub_anns_sim

            if num_sents > 0 and loss_evals >= num_sents:
                finish_flag = True
                break
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']

        if verbose:
            print('evaluating [%s] ... image[%d/%d]\'s sents, acc=%.2f%%, (%.4f), model time (per sent) is %.2fs' % \
                  (split, ix0, ix1, acc * 100.0 / loss_evals, loss, model_time / len(sent_ids)))

        model_time = 0
        our_model_time = 0
        DTMR_model_time = 0


        if finish_flag or data['bounds']['wrapped']:
            break
    npy_name = opt['dataset'] + '_' + split + '_sent_sub_anns_sim'
    np.save(npy_name + '.npy', Sub_anns_sim)

    npy_name2 = opt['dataset'] + '_' + split + '_proposal_spatial_scores'
    np.save(npy_name2 + '.npy', proposal_spatial_scores)
    print(len(proposal_spatial_scores))

    # print('sent_num is %d, avg model time (per sent) is %.5fs,'
    #       'avg our_model_time (per sent) is %.5fs, avg DTMR_model_time (per sent) is %.5fs' % \
    #       (loss_evals, avg_model_time /loss_evals, avg_our_model_time / loss_evals,
    #        avg_DTMR_model_time / loss_evals))
    # print('avg model time (per sent) is %.5fs,' % \
    #       (end1-start1))

    #print(loss_evals)
    #print( len(correct_sent_ids ))
    return loss_sum / loss_evals, acc / loss_evals, predictions
