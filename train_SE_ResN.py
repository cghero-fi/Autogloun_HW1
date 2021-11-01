from autogluon.vision import ImagePredictor, ImageDataset
import autogluon.core as ag
import csv
from tqdm import tqdm
from PIL import Image
import os
import pandas as pd
import math
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-name"
                    , type=str
                    , nargs='?'
                    , default = 'new_model'
                    , help='model name in ag_file')
args = parser.parse_args()
dataset = ImageDataset.from_folder('./train')

print(dataset)
print(type(dataset))

#model = ag.Categorical('resnest269', 'resnet152_v1d', 'se_resnext101_64x4d')
model = ag.Categorical('se_resnext101_64x4d')
# you may choose more than 70+ available model in the model zoo provided by GluonCV:
model_list = ImagePredictor.list_models()

batch_size = ag.Categorical(8, 16, 32)
lr = ag.Categorical(1e-2, 5*1e-2, 5*1e-3)


hyperparameters={'model': model, 'batch_size': batch_size, 'lr': lr, 'epochs': 300}
predictor = ImagePredictor(path='save_path')
predictor.fit(dataset, time_limit=60*60*3 , hyperparameters=hyperparameters 
    ,hyperparameter_tune_kwargs={'searcher': 'bayesopt', 'num_trials': 12, 'max_reward': 1.0} )
#predictor.fit(dataset, hyperparameters: {
#'model': Categorical('coat_lite_small', 'twins_pcpvt_base', 'swin_base_patch4_window7_224'), 'lr': Real(1e-5, 1e-2, log=True), 'batch_size': Categorical(8, 16, 32, 64, 128), 'epochs': 200, 'early_stop_patience': 50 },
#
#hyperparameter_tune_kwargs: {
#        'num_trials': 1024, 'searcher': 'random',
#        }, 'time_limit': 12*3600,)
print('Top-1 val acc: %.3f' % predictor.fit_summary()['valid_acc'])

filename = args.name + '.ag'
predictor.save( r'ag_file/' + filename)

# time_limit = 10 * 60 # 10mins
# predictor = ImagePredictor()
# predictor.fit(dataset, time_limit=time_limit)

# print('Top-1 val acc: %.3f' % predictor.fit_summary()['valid_acc'])
