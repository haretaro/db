import MeCab
import codecs
import json
import numpy as np
import os
import sys

def get(json_file, wakati=True):
    tagger = MeCab.Tagger('-O wakati')
    with codecs.open(json_file, 'r', 'utf-8') as f:
        j = json.load(f)
        utterances = []
        labels = []
        for u in j['turns']:
            label = [a['breakdown'] for a in u['annotations']]
            label = ''.join(label)
            if len(label) > 0:
                ratio = (label.count('O') / len(label),
                        label.count('T') / len(label),
                        label.count('X') / len(label))
            else:
                ratio = 0
            labels.append(ratio)
            utterance = tagger.parse(u['utterance']) if wakati else u['utterance']
            utterance = utterance.replace('\n', '<EOS>')
            utterances.append(utterance)
        return utterances, labels

def get_corpora_from(directory):
    corpora = []
    for _, _, files in os.walk(directory):
        for file_ in files:
            path = os.path.join(directory, file_)
            utterances, labels = get(path)
            corpora.append((utterances, labels))
    return corpora

def get_corpora():
    data = get_corpora_from('data/DCM')
    data += get_corpora_from('data/DIT')
    data += get_corpora_from('data/IRS')
    return data

def get_test_corpora():
    data = get_corpora_from('data/test/DCM')
    data += get_corpora_from('data/test/DIT')
    data += get_corpora_from('data/test/IRS')
    return data

def preprocess(corpora, word2index, UNK=0):
    xs, ys = [], []
    for corpus in corpora:
        x = []
        for i, (utterance, label) in enumerate(zip(*corpus)):
            if i == 0:
                continue
            elif i%2 == 1:
                x = utterance.split()
            else:
                x.extend(utterance.split())
                xs.append(x)
                ys.append(label)
    xs = [[word2index[w] if w in word2index else UNK for w in x ] for x in xs]
    xs = [np.asarray(x, dtype=np.int32) for x in xs]
    ys = np.asarray(ys, dtype=np.float32)
    return xs, ys

if __name__ == '__main__':
    data = get_corpora()
    for corpus in data:
        print('----')
        for utterance, label in zip(*corpus):
            print(utterance, label)
    print(len(data))
