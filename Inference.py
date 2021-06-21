# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:16:50 2020

@author: CITI
"""


#%%

import os
import torch
import numpy as np
import librosa
import pandas as pd
import soundfile as sf


from models.BaselineBLSTM import RNNDownBeatProc as bsl_blstm
from madmom.features.downbeats import DBNDownBeatTrackingProcessor as DownBproc
from madmom.features.downbeats import RNNDownBeatProcessor as RNNproc_api
import utils

from hmmopt import get_dlm_activation
from pathlib import Path


f_measure_threshold=0.07 # 70ms tolerance as set in paper
beats_per_bar = [3, 4]
global_sr = 44100

def get_wav(audio_file_path):
    wav = librosa.load(audio_file_path,
                sr= global_sr, )[0]
    return wav

def get_feature(audio_file_path):

    features = utils.madmom_feature(get_wav(audio_file_path))
    return features

def df2eval_dictlist(df, withMadmom = False):
    if withMadmom:
        eval_dictlist = [
                        {
            'model_type': 'madmom_api', # should allow using api for comparison 
            'model_dir': None, # only allow None when using madom_api
            'model_simpname': 'Madmom', 
            'n_tempi':  60, 
            'transition_lambda': 100, 
            'observation_lambda': 16, 
            'threshold': 0.05,
            
        }, ]
    else:
        eval_dictlist = []
    for model_ind in range(len(df)):
        model_dict={
                'model_type': df['model_type'].iloc[model_ind],
                'model_dir': df['model_dir'].iloc[model_ind], 
                'model_simpname': df['model_simpname'].iloc[model_ind], 
                'n_tempi': df['n_tempi'].iloc[model_ind], 
                'transition_lambda': df['transition_lambda'].iloc[model_ind], 
                'observation_lambda':df['observation_lambda'].iloc[model_ind], 
                'threshold': df['threshold'].iloc[model_ind],

                }
        eval_dictlist.append(model_dict)
    return eval_dictlist


def main():
    cuda_num = 0 
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    
    ## get model information from csv
    input_csv_name = 'HMMparamNote_0529.csv'
    input_csv_path = os.path.join('./hmm_optimization/merged_opt_results/', input_csv_name)

    df = pd.read_csv(input_csv_path)
    ## creating inputinfo_list for evaluation
    modelinfo_list = df2eval_dictlist(df, withMadmom =False)
    
    audio_file_path = './datasets/original/gtzan/audio/blues.00008.wav'
    for modelinfo in modelinfo_list:
        ### RNN init
        if modelinfo['model_type'] =='madmom_api':
            rnn = RNNproc_api()

        elif modelinfo['model_type'] =='bsl_blstm':
            rnn = bsl_blstm()
    

        else: 
            print("can't find model for :", modelinfo['model_dir'])
            
        if not modelinfo['model_type'] =='madmom_api':
            model_fn = 'RNNBeatProc.pth'
            model_path = os.path.join(modelinfo['model_dir'] , model_fn)
    
            state = torch.load(model_path, map_location = device)
            rnn.load_state_dict(state)
            if torch.cuda.is_available():
                rnn.cuda(device.index)
    
        ### DBN init
        hmm_proc = DownBproc(beats_per_bar = beats_per_bar, 
                         num_tempi = modelinfo['n_tempi'], 
                     transition_lambda = modelinfo['transition_lambda'], 
                     observation_lambda = modelinfo['observation_lambda'], 
                     threshold = modelinfo['threshold'], fps = 100)
        ### get feature of input audio file 
        feat = get_feature(audio_file_path)
        
        if modelinfo['model_type'] == 'madmom_api':
            activation  = rnn(audio_file_path)
        else:
            ### beat shape: (numof beats, 2)
            ### feat (feature) shape: (timeframes, 314 ), 
            activation = get_dlm_activation(rnn, device, feat)
            
            beat_fuser_est = hmm_proc( activation)
            txt_out_folder = os.path.join('./inference/out_txt', modelinfo['model_simpname'])
            if not os.path.exists(txt_out_folder):
                Path(txt_out_folder).mkdir(parents = True, exist_ok = True)
            txt_out_path = os.path.join(txt_out_folder, os.path.basename(audio_file_path)+'.beats')
            np.savetxt(txt_out_path, beat_fuser_est, fmt = '%.5f')
                
            
            # downbeat = beat_fuser_est[np.where(beat_fuser_est[:,1]==1), 0]
            beat = beat_fuser_est[:, 0]
            ori_wav = get_wav(audio_file_path)
            click = librosa.clicks(times = beat, sr = 44100, length = len(ori_wav))
            click_wav = ori_wav + click
            #librosa.output.write_wav(os.path.join(txt_out_folder, os.path.basename(audio_file_path)), click_wav, sr = 44100)
            sf.write(os.path.join(txt_out_folder, os.path.basename(audio_file_path)), click_wav, samplerate  = 44100)


if __name__ == "__main__":
    main()