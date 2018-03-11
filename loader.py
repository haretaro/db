import codecs
import json
import MeCab
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

if __name__ == '__main__':
    data = get_corpora()
    for corpus in data:
        print('----')
        for utterance, label in zip(*corpus):
            print(utterance[:-1], label)
    print(len(data))
