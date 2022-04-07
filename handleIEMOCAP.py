#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhu Wenjing
# Date: 2022-03-07
# E-mail: zhuwenjing02@duxiaoman.com

import glob
import json
import os

LABEL = {
    'neu': '01',  #: 'neutral',
    'fru': '02',  #: 'calm',
    'hap': '03',  #: 'happy',
    'sad': '04',  #: 'sad',
    'ang': '05',  #: 'angry',
    'fea': '06',  #: 'fearful',
    'exc': '07',  #: 'disgust',
    'sur': '08',  #: 'surprised'
    'xxx': '09',  #: 'other'
}

from path import IEMOCAP

PATH_TXT = glob.glob(IEMOCAP + "/*/dialog/EmoEvaluation/S*.txt")
PATH_WAV = glob.glob(IEMOCAP + "/*/sentences/wav/*/S*.wav")

PAIR = {}


def getPair():
    for path in PATH_TXT:
        with open(path, 'r') as f:
            fr = f.read().split("\t")
            for i in range(len(fr)):
                if (fr[i] in LABEL):
                    PAIR[fr[i - 1]] = fr[i]

def rename():
    for i in PATH_WAV:
        for j in PAIR:
            if (os.path.basename(i)[:-4] == j):
                k = j.split('_')
                if (len(k) == 3):
                    name = os.path.dirname(i) + '/' + k[0] + '-' + k[1] + '-' + LABEL[PAIR[j]] + '-01-' + k[2] + '.wav'
                    os.rename(src=i, dst=name)
                    #os.system(f'cp {i} data/{os.path.basename(name)}')
                    #print(name)
                    '''
                    Ses01F_impro01_F000.wav
                    k[0]:Ses01F
                    k[1]:impro01
                    k[2]:F000
                    Ses01F-impro01-XX-01-F000.wav
                    '''
                elif (len(k) == 4):
                    name = os.path.dirname(i) + '/' + k[0] + '-' + k[1] + '-' + LABEL[PAIR[j]] + '-01-' + k[2] + '_' + \
                           k[
                               3] + '.wav'
                    os.rename(src=i, dst=name)
                    #os.system(f'cp {i} data/{os.path.basename(name)}')
                    #print(name)
                    '''
                    Ses03M_script03_2_F032.wav
                    k[0]:Ses03M
                    k[1]:script03
                    k[2]:2
                    k[3]:F032
                    Ses03M-script03-XX-01-2_F032.wav
                    '''


if __name__ == '__main__':
    pairPath = IEMOCAP + "/pair.json"
    if (os.path.exists(pairPath)):
        with open(pairPath, 'r') as f:
            PAIR = json.load(f)
    else:
        getPair()
        with open(pairPath, 'w') as f:
            json.dump(obj=PAIR, fp=f)
    print('Starting...')
    rename()



