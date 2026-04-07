#提取所有训练句子中的名词种数
import os
import io
import sys
import json
import argparse
import os.path as osp
import numpy as np
import pdb

from tqdm import tqdm

train_sent_ids = np.load('D:/research/data/refcocog/refcocog_train_sent_ids.npy')
train_sent_ids = train_sent_ids.tolist()

vocab_NN = set()

sents = json.load(open(osp.join('D:/research/data/parsed_atts/refcocog_google', 'sents.json')))

cnt = 0
word2count = {}
word_count_threshold = 5
for sent in tqdm(sents):  # 每个句子
    sent_id = sent['sent_id']
    if sent_id in train_sent_ids:
        cnt = cnt+1
        words_info = sent['parse']['words']
        for word_info in words_info:  # 每个单词
            word = word_info[0]
            word2count[word] = word2count.get(word, 0) + 1
            word_tag = word_info[1]['PartOfSpeech']
            if word_tag == "NN":
                vocab_NN.add(word)
            # elif word_tag == "JJ":
                 #vocab_JJ.add(word)
            # elif word_tag == "JJS":
            #     vocab_JJS.add(word)

vocab_NN_thre = []

for wd in vocab_NN:
    if word2count[wd] > word_count_threshold:
        vocab_NN_thre.append(wd)

# vocab_file = 'D:/research/code/KPRN-master/KPRN-master/cache/word_embedding/vocabulary_72700.txt'
# f = open(vocab_file, "r")
# results = f.read().splitlines()  # list
# for wd in vocab_NN_thre:
#     if wd in results:
#         vocab_NN_glove.append(wd)

# for wd in vocab_JJ:
#     if word2count[wd] > word_count_threshold:
#         vocab_JJ_thre.append(wd)
#
# for wd in vocab_JJS:
#     if word2count[wd] > word_count_threshold:
#         vocab_JJS_thre.append(wd)
vocab_NN = list(vocab_NN)
# vocab_JJ = list(vocab_JJ)
# vocab_JJS = list(vocab_JJS)

json_file_path1 = 'D:/research/data/refcocog/refcocog_vocab_NN_thre.json'
json_file1 = open(json_file_path1, mode='w')
json.dump(vocab_NN_thre, json_file1, indent=4)



