# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:03:02 2020


@author: CITI
"""
#%%
import os
import glob
from pathlib import Path
import json
import pandas as pd

def getDoneDirs(out_dir):
    csv_list = [i for i in glob.glob(os.path.join(out_dir, "*.csv")) if 'bestParams_' in i]
    model_dirs = []
    for csvpath in csv_list:
#        break
        df = pd.read_csv(csvpath)
        model_dir = df['model_dir'].unique().tolist()
        model_dirs += model_dir
    return model_dirs

def main():
    ### you can modify the filename
    date = '0529'
    csv_name = 'hmmopt_todolist_'+date+'.csv'
    
    
    ### place to save optimization history    
    out_dir = './hmm_optimization/opt_history'
    if not os.path.exists(out_dir):
        Path(out_dir).mkdir(parents = True, exist_ok = True)
    
    ### collect paths of models that have found their best hmm parameters
    done_dirs = getDoneDirs(out_dir)
    ### collect paths of trained models from experiment folders
    exp_main_dir = './experiments'
    exp_model_dirs = glob.glob(os.path.join(exp_main_dir, "*"))
    
    
    model_dict_list = []
    for eachdir in exp_model_dirs:
        # break
    
        try:
            json_path = os.path.join(eachdir, 'RNNbeat.json')
            with open(json_path, 'r') as reader:
                jf = json.loads(reader.read())
        except:
            print("!!!!!=====>can't open json:", json_path)
            print("I'll just continue!!!")
            continue
        
        if eachdir in done_dirs:
            hmm = 'done'
        else:
            if int(jf['stop_t']) ==4 and not (int(jf['epochs_trained'])-int(jf['best_epoch']))%20:
                hmm = "ready"
            else:
                hmm = ''
        if 'model_info' in jf.keys():
            model_info = jf['model_info']
            model_dict = {
                    'hmm': hmm,
                    'model_dir': eachdir,
                    'best/trained epoch': '{}//{}'.format(str(jf['best_epoch']), jf['epochs_trained']), 
                    'lr_change_epoch': jf['lr_change_epoch'],
                    }
            model_dict.update(model_info)
            ### add version number for model_simpname
            version = os.path.basename(eachdir).split('_')[-2][-1]
            if not model_dict['model_simpname'].endswith('_v'+version):
                model_dict['model_simpname']=model_dict['model_simpname']+'_v'+version
        else:
            model_dict = {
                    'model_dir': eachdir, 
                    'model_type': '', 
                    'model_simpname': '',
                    'hmm':hmm,
                    }
        model_dict_list.append(model_dict)
    
    
    ft_notes_dir = './hmm_optimization/hmmopt_todolist'
    if not os.path.exists(ft_notes_dir):
        Path(ft_notes_dir).mkdir(parents = True, exist_ok = True)
    
    
    
    csv_spath = os.path.join(ft_notes_dir, csv_name)
    csv_df = pd.DataFrame(model_dict_list)
    csv_df['best/trained epoch'] = csv_df['best/trained epoch'].astype(str)
    csv_df.to_csv(csv_spath)


if __name__=="__main__":
    main()