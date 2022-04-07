# #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Zhu Wenjing
# @Email:  zhuwenjing02@duxiaoman.com
# @Date:   2022-03-09 10:04:58
# @Last Modified by:   Zhu Wenjing
# @Last Modified time: 2022-04-07 15:09:53
# -*- coding:utf-8 -*-
# Author: Zhu Wenjing
# Date: 2022-03-07
# E-mail: zhuwenjing02@duxiaoman.com

import os
import pickle
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from collections import Counter

import models
import data_loader
from path import datasets_path
from process import IEMOCAP_LABEL
from pytorch_model_summary import summary

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='...')
    # features
    parser.add_argument('-f', '--features_to_use',default='mfcc',type=str,help='{"mfcc" , "logfbank","fbank","spectrogram","melspectrogram"}')
    parser.add_argument('-i', '--impro_or_script',default='impro',type=str,help='select features')
    parser.add_argument('-s', '--sample_rate',default=16000,type=int,help='sample rate, default is 16000')
    parser.add_argument('-n', '--nmfcc',default=26,type=int,help='MFCC coefficients')
    parser.add_argument('--train_overlap',default=1.6,type=float,help='train dataset overlap')
    parser.add_argument('--test_overlap',default=1.6,type=float,help='test dataset overlap')
    parser.add_argument('--segment_length',default=1.8,type=float,help='segment length')
    parser.add_argument('--toSaveFeatures',default=False,type=bool,help='Save features')
    parser.add_argument('--loadFeatures',default=True,type=bool,help='load features')
    parser.add_argument('--featuresFileName',default=None,type=str,help='features file name')
    # model
    parser.add_argument('-m', '--model',default='GLAM',type=str,help='specify models')
    parser.add_argument('--head',default=4,type=int,help='head numbers')
    parser.add_argument('--attn_hidden',default=64,type=int,help='attention hidden size')
    parser.add_argument('--SaveModel',default=False,type=bool,help='Save model')
    # Datasets
    parser.add_argument('--split_rate',default=0.8,type=float,help='dataset split rate')
    parser.add_argument('--aug',default=None,type=str,help='augmentation')
    parser.add_argument('--padding',default=None,type=str,help='padding')
    # training
    parser.add_argument('--seed',default=987654,type=int,help='random seed')
    parser.add_argument('-b','--batch_size',default=32,type=int,help='batch size')
    parser.add_argument('-l','--learning_rate',default=0.001,type=float,help='learning rate')
    parser.add_argument('--lr_min',default=1e-6,type=float,help='minimum lr')
    parser.add_argument('--lr_schedule',default='exp',type=str,help='lr schedule')
    parser.add_argument('--optimizer',default='adam',type=str,help='optimizer')
    parser.add_argument('-e','--epochs',default=50,type=int,help='epochs')
    parser.add_argument('--iter',default=1,type=int,help='iterations')
    parser.add_argument('-g','--gpu',default=None,type=int,help='specify gpu device')
    parser.add_argument('--weight',default=None,type=bool)
    # parser.add_argument('--mixup',default=None,type=bool)
    parser.add_argument('--alpha',default=0.5,type=float, help='mixing up trick, using beta distribution')
    # config file
    parser.add_argument('-c', '--config',default=None,type=str,help='models')
    parser.add_argument('-d', '--datadir',default='data/',type=str,help='models')

    args = parser.parse_args()
    return args

def config(args):
    # GPU Config
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES']=''
    elif args.gpu is not None and args.gpu > -1:
        torch.cuda.set_device(args.gpu)

    # Trim 
    if args.datadir[-1] == '/': args.datadir = args.datadir[:-1]

    # get kwargs
    kws = vars(args)
    if args.config and os.path.exists(kws['config']):
        with open(kws['config']) as f:
            config_kws = json.load(f)
            for k, v in config_kws.items():
                if v: kws[k] = v
            # kws.update(config_kws)
    return kws

# Initial setting
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def mixup_data(x, y, alpha=0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if torch.cuda.is_available():
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    mixed_x, y_a, y_b = map(Variable,(mixed_x, y_a, y_b))
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_loss(model, criterion, x,y,alpha):
    if alpha > 0:
        mix_x, ya, yb, lam = mixup_data(x,y,alpha)
        if torch.cuda.is_available():
            mix_x = mix_x.cuda()
            ya = ya.cuda()
            yb = yb.cuda()
        ya = ya.squeeze(1)
        yb = yb.squeeze(1)

        out = model(mix_x.unsqueeze(1))
        loss = mixup_criterion(criterion, out, ya, yb, lam)
        return loss
    else:
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        y = y.squeeze(1)
        out = model(x.unsqueeze(1))
        loss = criterion(out,y)
        return loss

from torch.optim.lr_scheduler import _LRScheduler
import warnings
class WarmStepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, step_size, warm_step=20, gamma=0.1, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.warm_step = warm_step
        self.gamma = gamma
        super(WarmStepLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** ( 0 if self.last_epoch < self.warm_step else self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]


def train(kws):
    print(f'Model: {kws["model"]}')
    shape = train_X.shape[1:]
    print(shape)
    model = getattr(models, kws['model'])(shape = shape, **kws)
    print(summary(model,torch.zeros((1, 1, *shape))))

    if torch.cuda.is_available():
        model = model.cuda()

    if kws['weight']:
        count = Counter(train_y)
        nums = np.array([count['netral'], count['happy'], count['sad'], count['angry']])
        weight = torch.Tensor(1-nums/nums.sum())
        if torch.cuda.is_available():
            weight = weight.cuda()
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    if kws['optimizer'] == 'adam':        
        optimizer = optim.Adam(model.parameters(), lr=kws['learning_rate'],
                            weight_decay=1e-6)
    elif kws['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                    lr=kws['learning_rate'],
                    momentum=0.9,
                    weight_decay=1e-4,
                    nesterov=True)

    if kws['lr_schedule'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer,T_max=10)
    elif kws['lr_schedule'] == 'exp':
        scheduler = ExponentialLR(optimizer,0.95)
    elif kws['lr_schedule'] == 'step':
        scheduler = StepLR(optimizer, 10, 0.1)
    elif kws['lr_schedule'] == 'warmstep':
        scheduler = WarmStepLR(optimizer, 10, 0.1)
    else:
        scheduler = StepLR(optimizer, 10000, 1)

    print("training...")
    fh = open(f'{kws["datadir"]}/{kws["model"]}_train.log', 'a')
    maxACC = 0
    totalrunningTime = 0
    learning_rate = kws['learning_rate']
    MODEL_PATH = f'{kws["datadir"]}/model_{kws["model"]}_{filename}.pth'
    floss = open(f'{kws["dirname"]}/{kws["model"]}_{filename}.loss','a')
    floss.write(f'{kws["seed"]}\t')
    for i in range(kws['epochs']):
        startTime = time.perf_counter()
        # time.perf_counter
        tq = tqdm(total=len(train_y))
        model.train()
        print_loss = 0
        for _, data in enumerate(train_loader):
            x, y = data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            loss = get_loss(model, criterion, x, y, kws['alpha'])

            print_loss += loss.data.item() * kws['batch_size']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tq.update(kws['batch_size'])
        tq.close()
        if optimizer.param_groups[0]['lr'] >= kws['lr_min']: scheduler.step()
        floss.write(f'{print_loss / len(train_X)}\t')
        print('epoch: {}, lr: {:.4}, loss: {:.4}'.format(i, optimizer.param_groups[0]['lr'], print_loss / len(train_X)))
        fh.write('epoch: {}, lr: {:.4}, loss: {:.4}\n'.format(i, optimizer.param_groups[0]['lr'], print_loss / len(train_X)))
        # validation
        endTime = time.perf_counter()
        totalrunningTime += endTime - startTime
        fh.write(f'{totalrunningTime}\n')
        model.eval()
        y_true, y_pred = [], []
        for val in val_dict:
            x, y = val['X'], val['y']
            x = torch.from_numpy(x).float()
            y_true.append(y)
            if torch.cuda.is_available():
                x = x.cuda()
            if (x.size(0) == 1):
                x = torch.cat((x, x), 0)
            out = model(x.unsqueeze(1))
            pred = out.mean(dim=0)
            pred = torch.max(pred, 0)[1].cpu().numpy()
            y_pred.append(int(pred))
        floss.write('\n')

        report = classification_report(y_true, y_pred,digits=6, target_names=target_names)
        report_dict = classification_report(y_true, y_pred, digits=6, target_names=target_names, output_dict=True)
        matrix = metrics.confusion_matrix(y_true, y_pred)

        WA = report_dict['accuracy'] * 100
        UA = report_dict['macro avg']['recall'] * 100
        macro_f1 = report_dict['macro avg']['f1-score'] * 100
        w_f1 = report_dict['weighted avg']['f1-score'] * 100
        
        ACC = (WA + UA)/2
        if maxACC < ACC: 
            maxACC, WA_, UA_ = ACC, WA, UA
            macro_f1_, w_f1_ = macro_f1, w_f1
            best_re, best_ma = report, matrix
            if kws["SaveModel"]: torch.save(model.state_dict(), MODEL_PATH)
        print(report)
        print(matrix)
        print('The best result ----------')
        print(f'WA: {WA_:.4f}%, UA: {UA_:.4f}%, macro f1: {macro_f1_:.4f}%, weighted f1: {w_f1_:.4f}%')
        print('--------------------------')

        fh.write(report)
        fh.write('\n')
        fh.write(f'{matrix}')
        fh.write('\n')
        fh.write('The best result ----------')
        fh.write('\n')
        fh.write(f'WA: {WA_:.4f}%, UA: {UA_:.4f}%, macro f1: {macro_f1_:.4f}%, weighted f1: {w_f1_:.4f}%')
        fh.write('\n')
        fh.write('--------------------------')
        fh.write('\n')
    
    del model

    return maxACC, WA_, UA_, macro_f1_, w_f1_, best_re, best_ma

if __name__ == '__main__':
    from sklearn.metrics import classification_report
    from sklearn import metrics
    from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR

    args = parse_args()
    kws = config(args)

    setup_seed(kws['seed'])

    if kws['featuresFileName'] or kws['config']:
        surffix = '.same' 
    else:
        surffix = '.metric'

    filename = f'{kws["features_to_use"]}_{kws["impro_or_script"]}'
    if kws['featuresFileName'] is None:
        kws['featuresFileName'] = f'{kws["datadir"]}/features_{filename}.pkl'    

    if not os.path.exists(kws["datadir"]): os.system(f'mkdir -p {kws["datadir"]}')

    with open(kws["datadir"] + '/last.json','w') as f:
        del kws['config']
        json.dump(kws,f,indent=4)

    if os.path.dirname(kws["datadir"]): 
        kws["dirname"] = os.path.dirname(kws["datadir"])
    else:
        kws["dirname"] = kws["datadir"]

    print('Load features...')
    if kws['loadFeatures'] and os.path.exists(kws['featuresFileName']):
        with open(kws['featuresFileName'], 'rb')as f:
            features = pickle.load(f)
        train_X, train_y, val_dict, info = features['train_X'], features['train_y'], features['val_dict'], features.get('info','')
    else:
        from process import process_IEMOCAP
        train_X, train_y, val_dict, info = process_IEMOCAP(datasets_path, IEMOCAP_LABEL, **kws)

    train_data = data_loader.DataSet(train_X, train_y)
    train_loader = DataLoader(train_data, batch_size=kws['batch_size'], shuffle=True)

    target_names = ['neutral','happy','sad','angry']

    best = [0]
    t0 = time.perf_counter()
    for _ in range(kws['iter']):
        best_ = train(kws)
        if best and best_[0] > best[0]:
            best = best_

    maxACC, WA_, UA_, macro_f1_, w_f1_, best_re, best_ma = best
    facc = f'{kws["dirname"]}/{kws["model"]}_{filename}{surffix}'
    with open(facc, 'a+') as f:
        f.write(f'{kws["seed"]:16}\t{maxACC:.4f}\t')
        f.write(f'{WA_}\t{UA_}\t')
        f.write(f'{macro_f1_}\t{w_f1_}\n')
    flog = f'{kws["dirname"]}/{kws["model"]}_{filename}.log'
    with open(flog, 'a+') as f:
        f.write(f'-------------------------------------------\n')
        f.write(f'{kws["seed"]}\n')
        f.write(best_re)
        f.write('\n')
        f.write(f'{best_ma}')
        f.write('\n')
        f.write(f'{info}')
        f.write('\n')
        f.write(f'Running time with {kws["iter"]} iterations: {time.perf_counter() - t0}\n')
        f.write(f'-------------------------------------------\n')
        f.write('\n')
