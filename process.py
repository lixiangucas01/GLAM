#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhu Wenjing
# Date: 2022-03-07
# E-mail: zhuwenjing02@duxiaoman.com

import os
import glob
from tqdm import tqdm
import librosa
import numpy as np
import argparse
import pickle
import math
from collections import Counter
import random
import json
from python_speech_features import logfbank, fbank, sigproc

from path import datasets_path

class FeatureExtractor(object):
    def __init__(self, sample_rate, nmfcc = 26):
        self.sample_rate = sample_rate
        self.nmfcc = nmfcc

    def get_features(self, features_to_use, X):
        X_features = None
        accepted_features_to_use = ("logfbank", 'mfcc', 'fbank', 'melspectrogram', 'spectrogram', 'interspeech2018')
        if features_to_use not in accepted_features_to_use:
            raise NotImplementedError("{} not in {}!".format(features_to_use, accepted_features_to_use))
        if features_to_use in ('logfbank'):
            X_features = self.get_logfbank(X)
        if features_to_use in ('mfcc'):
            X_features = self.get_mfcc(X,self.nmfcc)
        if features_to_use in ('fbank'):
            X_features = self.get_fbank(X)
        if features_to_use in ('melspectrogram'):
            X_features = self.get_melspectrogram(X)
        if features_to_use in ('spectrogram'):
            X_features = self.get_spectrogram(X)
        if features_to_use in ('interspeech2018'):
            X_features = self.get_spectrogram_interspeech2018(X)
        return X_features

    def get_logfbank(self, X):
        def _get_logfbank(x):
            out = logfbank(signal=x, samplerate=self.sample_rate, winlen=0.040, winstep=0.010, nfft=1024, highfreq=4000,
                           nfilt=40)
            return out

        X_features = np.apply_along_axis(_get_logfbank, 1, X)
        return X_features

    def get_mfcc(self, X, nmfcc=13):
        def _get_mfcc(x):
            mfcc_data = librosa.feature.mfcc(x, sr=self.sample_rate, n_mfcc=nmfcc)
            return mfcc_data

        X_features = np.apply_along_axis(_get_mfcc, 1, X)
        return X_features

    def get_fbank(self, X):
        def _get_fbank(x):
            out, _ = fbank(signal=x, samplerate=self.sample_rate, winlen=0.040, winstep=0.010, nfft=1024)
            return out

        X_features = np.apply_along_axis(_get_fbank, 1, X)
        return X_features

    def get_melspectrogram(self, X):
        # def _get_melspectrogram(x):
        #     mel = librosa.feature.melspectrogram(y=x, sr=self.sample_rate, n_fft=800, hop_length=400)[np.newaxis, :]
        #     delta = librosa.feature.delta(mel)
        #     delta_delta = librosa.feature.delta(delta)
        #     out = np.concatenate((mel, delta, delta_delta))
        #     return out
        def _get_melspectrogram(x):
            mel = librosa.feature.melspectrogram(y=x, sr=self.sample_rate, n_fft=800, hop_length=400)[np.newaxis, :]
            out=np.log10(mel).squeeze()
            return out

        X_features = np.apply_along_axis(_get_melspectrogram, 1, X)
        return X_features

    def get_spectrogram(self, X):
        def _get_spectrogram(x):
            frames = sigproc.framesig(x, 640, 160)
            out = sigproc.logpowspec(frames, NFFT=3198)
            out = out.swapaxes(0, 1)
            return out[:][:400]

        X_features = np.apply_along_axis(_get_spectrogram, 1, X)
        return X_features

    def get_spectrogram_interspeech2018(self, X):
        def _get_spectrogram(x):
            frames = sigproc.framesig(x, 640, 160)
            out = sigproc.magspec(frames, NFFT=3198)
            out = out / out.max() * 2 - 1 
            out = np.sign(out) * np.log(1+255*np.abs(out))/np.log(256)
            out = out.swapaxes(0, 1)
            return out[:][:400]

        X_features = np.apply_along_axis(_get_spectrogram, 1, X)
        return X_features

def segment(wavfile, 
            sample_rate = 16000,
            segment_length = 2,
            overlap = 1,
            padding = None):
    if isinstance(wavfile, str):
        wav_data, _ = librosa.load(wavfile, sr=sample_rate)
    elif isinstance(wavfile, np.ndarray):
        wav_data = wavfile
    else:
        raise f'Type type(wavfile) is not supported.'
    X= []
    seg_wav_len = segment_length * sample_rate
    wav_len = len(wav_data)
    if (seg_wav_len > wav_len): 
        if padding:
            n = math.ceil(seg_wav_len/wav_len)
            wav_data = np.hstack(n*[wav_data])
        else:
            return None, None
    index = 0
    while (index + seg_wav_len <= wav_len):
        X.append(wav_data[int(index):int(index + seg_wav_len)])
        assert segment_length - overlap > 0
        index += int((segment_length - overlap) * sample_rate)

    X = np.array(X)
    return X

def process(wavfiles, 
            labels,
            num_label = None,
            features_to_use = 'mfcc',
            sample_rate=16000,
            nmfcc = 26, 
            train_overlap=1, 
            test_overlap=1.6, 
            segment_length=2,
            split_rate = 0.8,
            featuresFileName = 'features.pkl',
            toSaveFeatures = True,
            aug=None,
            padding=None):

    # fnouse = open(featuresFileName.rsplit('.', 1)[0] + '.nouse', 'w+')
    # Split datatset
    n = len(wavfiles)
    train_indices = list(np.random.choice(range(n), int(n * split_rate), replace=False))
    valid_indices = list(set(range(n)) - set(train_indices))
    train_files = [(wavfiles[i], labels[i]) for i in train_indices]
    valid_files = [(wavfiles[i], labels[i]) for i in valid_indices]

    get_label = lambda x: x[1]
    train_info = json.dumps(Counter(map(get_label, train_files)))
    test_info = json.dumps(Counter(map(get_label, valid_files)))
    info = {'train': train_info, 'test': test_info}

    if num_label is not None:
        print(f'Amount of categories: {num_label}')
    print(f'Training Datasets: {len(train_files)}, Testing Datasets: {len(valid_files)}')

    if aug == 'upsampling':
        label_wav = {
            'neutral': [],
            'happy': [],
            'sad': [],
            'angry': [],
            }
        for wavfile, label in train_files:
            label_wav[label].append(wavfile)
        maxval = 0
        for l, w in label_wav.items():
            maxval = max(maxval, len(w))
        for l, w in label_wav.items():
            nw = len(w)
            indices = list(np.random.choice(range(nw), maxval - nw, replace=True))
            for i in indices:
                train_files.append((w[i],l))
        random.shuffle(train_files)
        print(f'After Augmentation...\nTraining Datasets: {len(train_files)}, Testing Datasets: {len(valid_files)}')

    feature_extractor = FeatureExtractor(sample_rate, nmfcc)

    print('Extracting features for training datasets')
    train_X, train_y = [], []
    for i, wavfile_label in enumerate(tqdm(train_files)):
        wavfile, label = wavfile_label
        X1= segment(wavfile, 
                    sample_rate = sample_rate, 
                    segment_length = segment_length,
                    overlap = train_overlap,
                    padding = padding)
        y1 = len(X1) * [label]
        if X1 is None: 
            # fnouse.write(f'train: {wavfile}\t{label}\n')
            continue
        X1 = feature_extractor.get_features(features_to_use, X1)
        train_X.append(X1)
        train_y += (y1)
    train_X = np.row_stack(train_X)
    # train_X = feature_extractor.get_features(features_to_use, train_X)
    print(f'Amount of categories after segmentation(training): {Counter(train_y).items()}')
    train_y = np.array(train_y)
    assert len(train_X) == len(train_y), "X length and y length must match! X shape: {}, y length: {}".format(train_X.shape, train_y.shape)

    print('Extracting features for test datasets')
    val_dict = []
    test_y = []
    if (test_overlap >= segment_length): test_overlap = segment_length / 2
    for i, wavfile_label in enumerate(tqdm(valid_files)):
        wavfile, label = wavfile_label
        X1 = segment(wavfile, 
                    sample_rate = sample_rate, 
                    segment_length = segment_length,
                    overlap = test_overlap, 
                    padding = padding)
        if X1 is None: 
            # fnouse.write(f'train: {wavfile}\t{label}\n')
            continue
        X1 = feature_extractor.get_features(features_to_use, X1)
        val_dict.append({
            'X': X1,
            'y': label,
            # 'path': wavfile
        })
        test_y += [label]
    print(f'Amount of categories after segmentation(test): {Counter(test_y).items()}')

    info['train_seg'] = f'{Counter(train_y).items()}'
    if (toSaveFeatures == True):
        print(f'Saving features to {featuresFileName}.')
        features = {'train_X': train_X, 'train_y': train_y,
                    'val_dict': val_dict, 'info': info}
        with open(featuresFileName, 'wb') as f:
            pickle.dump(features, f)

    return train_X, train_y, val_dict, info

def process_IEMOCAP(datasets_path,
        LABEL_DICT, 
        datadir = 'data/',
        featuresFileName = None,
        features_to_use = 'mfcc',
        impro_or_script='impro',
        sample_rate=16000, 
        nmfcc = 26,
        train_overlap=1, 
        test_overlap=1.6, 
        segment_length=2,
        split_rate = 0.8,
        toSaveFeatures = True,
        aug = None,
        padding = None,
        **kwargs):
    if not os.path.exists(datadir): os.system(f'mkdir -p {datadir}')
    num_label = {}
    if featuresFileName is None:
        featuresFileName = f'{datadir}/features_{features_to_use}_{impro_or_script}.pkl'
    
    if os.path.exists(datasets_path):
        wavdirname = datasets_path + '/*/sentences/wav/*/S*.wav'
        allfiles = glob.glob(wavdirname)
    else:
        raise (f'{datasets_path} not existed.')

    wavfiles, labels = [], []
    for wavfile in allfiles:
        if len(os.path.basename(wavfile).split('-'))<5: continue
        label = str(os.path.basename(wavfile).split('-')[2])
        if label not in LABEL_DICT: continue
        if impro_or_script != 'all' and (impro_or_script not in wavfile): continue
        wav_data, _ = librosa.load(wavfile, sr = sample_rate)
        seg_wav_len = segment_length * sample_rate
        wav_len = len(wav_data)
        if seg_wav_len > wav_len:
            if padding:
                n = math.ceil(seg_wav_len/wav_len)
                wav_data = np.hstack(n*[wav_data])
            else:
                continue

        label = LABEL_DICT[label]
        wavfiles.append(wav_data)
        labels.append(label)
        num_label[label] = num_label.get(label, 0) + 1

    train_X, train_y, val_dict, info = process(wavfiles, 
            labels,
            num_label = num_label,
            features_to_use = features_to_use,
            sample_rate=sample_rate, 
            nmfcc=nmfcc, 
            train_overlap=train_overlap, 
            test_overlap=test_overlap, 
            segment_length=segment_length,
            split_rate = split_rate,
            featuresFileName = featuresFileName,
            toSaveFeatures = toSaveFeatures,
            aug=aug,
            padding=padding)
    return train_X, train_y, val_dict, info

IEMOCAP_LABEL = {
    '01': 0,
    # '02': 'frustration',
    # '03': 'happy',
    '04': 1,
    '05': 2,
    # '06': 'fearful',
    '07': 3,  # excitement->happy
    # '08': 'surprised'
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing datasets')
    parser.add_argument('-d', '--datasets_path',default=datasets_path,type=str,help='models')
    parser.add_argument('--datadir',default='features',type=str)
    parser.add_argument('-b', '--batch',default=None,type=int,help='models')
    
    args = parser.parse_args()

    datadir = args.datadir,
    if args.batch: 
        features_to_use = 'mfcc'
        impro_or_script='impro'
        for i in range(args.batch):
            featuresFileName = f'{datadir[0]}/features_{features_to_use}_{impro_or_script}_{i}.pkl'
            # print(featuresFileName)
            train_X, train_y, val_dict = process_IEMOCAP(args.datasets_path, IEMOCAP_LABEL,
                datadir = args.datadir,
                features_to_use = 'mfcc',
                impro_or_script='impro',
                featuresFileName=featuresFileName,
                sample_rate=16000, 
                nmfcc = 26,
                train_overlap=1, 
                test_overlap=1.6, 
                segment_length=2,
                split_rate = 0.8,
                toSaveFeatures = True)
    else:
        train_X, train_y, val_dict = process_IEMOCAP(args.datasets_path, IEMOCAP_LABEL,
            datadir = args.datadir,
            features_to_use = 'mfcc',
            impro_or_script='impro',
            sample_rate=16000, 
            nmfcc = 26,
            train_overlap=1, 
            test_overlap=1.6, 
            segment_length=2,
            split_rate = 0.8,
            toSaveFeatures = True)
