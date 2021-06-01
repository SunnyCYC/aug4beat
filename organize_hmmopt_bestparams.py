# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 21:14:41 2020

@author: sunny
"""

#%%
import os
import numpy as np
import glob
from pathlib import Path
import pandas as pd

def getBestParams(df):
    ### best line is the first line
    n_tempi = df['n_tempi'].iloc[0]
    observation_lambda = df['observation_lambda'].iloc[0]
    transition_lambda = df['transition_lambda'].iloc[0]
    threshold = df['threshold'].iloc[0]
    
    fuser_f_sum = df['fuser_f sum'].iloc[0]
    fuser_beat_F_score = df['fuser_beat F-score'].iloc[0]
    fuser_downbeat_F_score = df['fuser_downbeat F-score'].iloc[0]

    model_dir = df['model_dir'].iloc[0]
    model_setting = df['model_setting'].iloc[0]
    if 'model_type' in df.keys():
        model_type = df['model_type'].iloc[0]
    else:
        model_type = None
    
    if 'model_simpname' in df.keys():
        version = os.path.basename(model_dir).split('_')[-2][-1]
        model_simpname = df['model_simpname'].iloc[0]
        if not model_simpname.endswith('_v'+version):
            model_simpname =model_simpname+'_v'+version
        
    else:
        model_simpname = None
    
    infor_dict = {
            'n_tempi': n_tempi, 
            'observation_lambda': observation_lambda, 
            'transition_lambda': transition_lambda, 
            'threshold': threshold, 
            
            'fuser_f sum': fuser_f_sum, 
            'fuser_beat F-score': fuser_beat_F_score, 
            'fuser_downbeat F-score': fuser_downbeat_F_score, 
            
            'model_dir': model_dir,
            'model_type': model_type,
            'model_simpname': model_simpname,
            'model_setting': model_setting,
            }
    return infor_dict


def get_ft_csv_list(params_dirs):
    merged_list = []
    for params_dir in params_dirs:

        if not os.path.exists(params_dir):
            print("======!!!!!!params_dir not exists:", params_dir)
        ft_csv_list = [i for i in glob.glob(os.path.join(params_dir, "*.csv")) if '_bestParams' in i ]
        merged_list +=ft_csv_list
    return merged_list



def main():
    ### input information here:
    output_dir = os.path.join('./hmm_optimization/', 'merged_opt_results')
    date = '0529'
    output_csv_fn = 'HMMparamNote_'+date + '.csv'
    csv_outputpath = os.path.join(output_dir, output_csv_fn)
    
    ### create output dir
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents = True, exist_ok = True)    
    
    
    ### collect all hmm_opt notes
    ft_notes_maindir= './hmm_optimization/opt_history/'
    ft_csv_list = get_ft_csv_list([ft_notes_maindir])
    
    all_infor = []
    for csvfile in ft_csv_list:
    #    break
        csv_basename = os.path.basename(csvfile)
        df = pd.read_csv(csvfile, index_col = 0)
        infor_dict = getBestParams(df)
        infor_dict['csv_name'] = csv_basename
        all_infor.append(infor_dict)
    #    print(infor_dict)

    info_df = pd.DataFrame(all_infor)
    # for ind in range(len(info_df)):
    #     print(info_df.iloc[ind])
    
    info_df.to_csv(csv_outputpath)

if __name__ =="__main__":
    main()