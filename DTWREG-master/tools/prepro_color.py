# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
#每个句子的主语、宾语、关系词在
1、category name, 2、color, 3、size, 4、absolute location, 5、relative location, 
6、relative object, 7、and generic attribute
json.dump({
           "sent_sub_wordid":sent_sub_wordid,
           "sent_obj_wordid":sent_obj_wordid,
           "sent_rel_wordid":sent_rel_wordid,
           "sent_sub_classwordid": sent_sub_classwordid},
          open(osp.join('cache/sub_obj_wds', dataset + '_' + splitBy, "sent_extract.json"), 'w'))
'''

import os
import io
import sys
import json
import argparse
import os.path as osp
import numpy as np
from Config import *
import pdb

"""
sent_id: the id of each sentence (142210 totally, 127696 with subject)
word_id: the id of each word in the word list (72740 totally)
word_str: the content of each word
sub_id: the id of each candidate subject
sub_vec: the vector of each candidate subject

sentid_wordid: dictionary, sent_id : word_id
subid_wordid: list, sub_id : word_id
subid_subvec: list, sub_id : sub_vec
sentid_subid: dictionary, sent_id : sub_id
"""

forbidden_att = ['none', 'other', 'sorry', 'pic', 'extreme', 'rightest', 'tie', 'leftest', 'hard', 'only',
                 'darkest', 'foremost', 'topmost', 'leftish', 'utmost', 'lemon', 'good', 'hot', 'more', 'least', 'less',
                 'cant', 'only', 'opposite', 'upright', 'lightest', 'single', 'touching', 'bad', 'main', 'remote',
                 '3pm',
                 'same', 'bottom', 'middle']
forbidden_verb = ['none', 'look', 'be', 'see', 'have', 'head', 'show', 'strip', 'get', 'turn', 'wear',
                  'reach', 'get', 'cross', 'turn', 'point', 'take', 'color', 'handle', 'cover', 'blur', 'close', 'say',
                  'go',
                  'dude', 'do', 'let', 'think', 'top', 'head', 'take', 'that', 'say', 'carry', 'man', 'come', 'check',
                  'stuff',
                  'pattern', 'use', 'light', 'follow', 'rest', 'watch', 'make', 'stop', 'arm', 'try', 'want', 'count',
                  'lead',
                  'know', 'mean', 'lap', 'moniter', 'dot', 'set', 'cant', 'serve', 'surround', 'isnt', 'give', 'click']
forbidden_noun = ['none', 'picture', 'pic', 'screen', 'background', 'camera', 'edge', 'standing', 'thing',
                  'holding', 'end', 'view', 'bottom', 'center', 'row', 'piece', 'right', 'left']


# load vocabulary file
def load_vocab_dict_from_file(dict_file):
    if (sys.version_info > (3, 0)):
        with open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
    else:
        with io.open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
    vocab_dict = {words[n]: n for n in range(len(words))}
    return vocab_dict


def get_sub_obj_rel(vocab_dict, params):
    #sub_obj_nouns = set()
    #sub_obj_nouns.add('<unk>')
    noun_cnt = {}
    noun_cnt['<unk>'] = 10

    sents = json.load(open(osp.join('D:/research/data/parsed_atts/refcoco+_unc', 'sents.json')))
    sent_to_color_id = {}



    for sent in sents:

        sent_id = sent['sent_id']
        atts = sent['atts']

        # r1 = atts['r1'][0]
        #
        # r3 = atts['r3'][0]
        # r4 = atts['r4'][0]
        # r5 = atts['r5'][0]
        # r6 = atts['r6'][0]
        r2 = atts['r2'][0]
        r8_list = atts['r8']

        color_id = 3

        if r2 != 'none':
            if r2 in vocab_dict:

                color_id = vocab_dict[r2]

        elif len(r8_list) > 0:
            text_list = ["brown", "yellow", "pink", "purple", "black", "white", "red", "blue", "dark", "gray", "green", "orange" ]


            for x in r8_list:
                if x in text_list:
                    if x in vocab_dict:

                        color_id = vocab_dict[x]
                        break
        sent_to_color_id[sent_id] = color_id


    print( len(sent_to_color_id) )

    return sent_to_color_id


def main(params):
    # dataset_splitBy
    #data_root, dataset, splitBy = params['data_root'], params['dataset'], params['splitBy']

    # mkdir and write json file
    # if not osp.isdir(osp.join('cache/sent_color_wordid', dataset + '_' + splitBy)):
    #     os.makedirs(osp.join('cache/sent_color_wordid', dataset + '_' + splitBy))

    vocab_file = 'D:/research/code/EARN-main/EARN-main/cache/word_embedding/vocabulary_72700.txt'
    vocab_dict = load_vocab_dict_from_file(vocab_file)

    sent_to_color_id = get_sub_obj_rel(vocab_dict, params)

    json.dump({"sent_to_color_id": sent_to_color_id},
              open(osp.join('D:/research/data/refcoco+', "refcoco+_sent_to_color_id.json"), 'w'))

    print('related data have been written!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_json', default='sub_obj_wds.json', help='output json file')
    parser.add_argument('--data_root', default='data', type=str,
                        help='data folder containing images and four datasets.')
    parser.add_argument('--dataset', default='refcoco', type=str, help='refcoco/refcoco+/refcocog')
    parser.add_argument('--splitBy', default='unc', type=str, help='unc/google')
    parser.add_argument('--images_root', default='', help='root location in which images are stored')

    # argparse
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))

    # call main
    main(params)
