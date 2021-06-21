# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:00:00 2020

@author: CITI
"""
#%%
import os
import numpy as  np
from models.BaselineBLSTM import RNNDownBeatProc as bsl_blstm
from downbeat_dataset import EvalDataset 
from multiprocessing import Pool

import tqdm
import torch
from pathlib import Path
import pandas as pd
import mir_eval.util as mir_util
from scipy.special import softmax
from madmom.features.downbeats import DBNDownBeatTrackingProcessor as DownBproc


fps = 100
f_measure_threshold=0.07 # 70ms tolerance as set in paper
beats_per_bar = [3, 4]


def beat_eval(beat_label, beat_est, withdownbeat = True):
    if type(beat_est) ==list and len(beat_est)==0: ###1116: added for old models
        return []
    else:
        bmatching = mir_util.match_events(beat_label[:, 0], beat_est[:, 0], f_measure_threshold)
        b_reflen = beat_label.shape[0]
        b_estlen = beat_est.shape[0]
    
        # downbeat evaluation
        if withdownbeat:
            downbeat_label = beat_label[:,0][np.where(beat_label[:,1]==1)]
            downbeat_est = beat_est[:, 0][np.where(beat_est[:,1]==1)]
            dbmatching = mir_util.match_events(downbeat_label, downbeat_est, f_measure_threshold)
            db_reflen = len(downbeat_label)
            db_estlen = len(downbeat_est)
            
            return len(bmatching), b_reflen, b_estlen, len(dbmatching), db_reflen, db_estlen
        else:
            return len(bmatching), b_reflen, b_estlen 
    

def res2fmeasure(all_results, head = 'fuser'):
    
    ### 1116: added to deal with empty head for old models
    all_results_array = np.array(all_results)
    if all_results_array.shape[1]>0:
        beat_precision = float( all_results_array[:, 0].sum()/all_results_array[:, 2].sum())
        beat_recall = float( all_results_array[:, 0].sum()/all_results_array[:, 1].sum())
        beat_fmeasure = mir_util.f_measure(beat_precision, beat_recall)
        # downbeat
        dbeat_precision = float( all_results_array[:, 3].sum()/ all_results_array[:, 5 ].sum())
        dbeat_recall = float( all_results_array[:, 3].sum()/ all_results_array[:, 4].sum())
        dbeat_fmeasure = mir_util.f_measure(dbeat_precision, dbeat_recall)
        
        result_dict = {
                    head+'_f sum': beat_fmeasure + dbeat_fmeasure,
                    head+'_beat F-score': beat_fmeasure, 
                    head+'_beat_precision': beat_precision, 
                    head+'_beat_recall': beat_recall, 
                    head+'_downbeat F-score': dbeat_fmeasure, 
                    head+'_downbeat_precision': dbeat_precision, 
                    head+'_downbeat_recall': dbeat_recall, 
                    
                    }
    else:
        result_dict = {
                    head+'_f sum': None,
                    head+'_beat F-score': None, 
                    head+'_beat_precision': None, 
                    head+'_beat_recall': None, 
                    head+'_downbeat F-score': None, 
                    head+'_downbeat_precision': None, 
                    head+'_downbeat_recall': None, 
                    
                    }
            
    
    return result_dict

def process_worker(arg_list):

    param_dict, beat_list, activation_list = arg_list
    
    hmm_proc = DownBproc(beats_per_bar = beats_per_bar, num_tempi = param_dict['ntempi'], 
                         transition_lambda = param_dict['tlambda'], 
                         observation_lambda = param_dict['olambda'], 
                         threshold = param_dict['thres'], fps = fps)
    fuser_results = []
    mix_results = []
    nodrum_results = []
    drum_results = []
    for ind, activation in enumerate(activation_list):
#        break
        if type(activation) ==list:
            if len(activation) == 4:
            ### For drum-aware ensemble models:
                (fuser_activation, mix_activation, nodrum_activation, drum_activation) = activation
                beat_fuser_est = hmm_proc( fuser_activation)
                beat_mix_est = hmm_proc( mix_activation)
                beat_nodrum_est = hmm_proc( nodrum_activation)
                beat_drum_est = hmm_proc( drum_activation)
            else:
                print("unexpected len of activation")
        else:
            fuser_activation = activation
            beat_fuser_est = hmm_proc( fuser_activation)
            beat_mix_est = [] # not implemented for old models
            beat_nodrum_est = [] # not implemented for old models
            beat_drum_est = [] # not implemented for old models
            
        beat = beat_list[ind]
        fuser_results.append(beat_eval(beat, beat_fuser_est))
        mix_results.append(beat_eval(beat, beat_mix_est))
        nodrum_results.append(beat_eval(beat, beat_nodrum_est))
        drum_results.append(beat_eval(beat, beat_drum_est))
        
    fuser_dict = res2fmeasure(fuser_results, head = 'fuser')
    mix_dict = res2fmeasure(mix_results, head = 'mix')
    nodrum_dict = res2fmeasure(nodrum_results, head = 'nodrum')
    drum_dict = res2fmeasure(drum_results, head= 'drum')
     
    result_dict = {    
                'n_tempi':  param_dict['ntempi'], 
                'transition_lambda': param_dict['tlambda'], 
                'observation_lambda': param_dict['olambda'], 
                'threshold': param_dict['thres'],
                }
    result_dict.update(fuser_dict)
    result_dict.update(mix_dict)
    result_dict.update(nodrum_dict)
    result_dict.update(drum_dict)
    return result_dict 

def prediction_conversion(prediction):
    if len(prediction.shape) == 2:
        prediction = prediction.unsqueeze(0)
    pred_arr = prediction.detach().cpu().numpy()
    pred_acti = softmax(pred_arr, axis = 2)
    pred_acti = pred_acti.squeeze()
    
    model_activation = np.zeros((pred_acti.shape[0], 2))
    model_activation[:, 0] = pred_acti[:, 2] # beat class
    model_activation[:, 1] = pred_acti[:, 1] # downbeat class
    return model_activation

def get_dlm_activation(rnn, device, np_2dfeature):
    """ get deep learning model activations"""
    input_feature = torch.tensor(np_2dfeature[np.newaxis, :, :]).float().to(device)
    rnn.eval()
    
    with torch.no_grad():
        activation = rnn(input_feature)
    
    ### drum-aware models 
    if type(activation)==tuple and len(activation) ==6:
        beat_fused, beat_mix, beat_nodrum, beat_drum, x_nodrum_hat, x_drum_hat = activation
        fuser_activation = prediction_conversion(beat_fused)
        mix_activation = prediction_conversion(beat_mix)
        nodrum_activation = prediction_conversion(beat_nodrum)
        drum_activation = prediction_conversion(beat_drum)
        model_activation = [fuser_activation, mix_activation, nodrum_activation, drum_activation]
        
        return model_activation

    else:
        beat_fused = activation 
        fuser_activation = prediction_conversion(beat_fused)
        
        return fuser_activation
    
def getValidtxt_paths(datasets_list, main_dataset_dir, valid_fn = 'valid_audiofiles.txt'):
    validtxt_paths = []
    for eachdataset in datasets_list:
#        break
        valid_txt_path = os.path.join(main_dataset_dir, eachdataset, valid_fn)
        if os.path.exists(valid_txt_path):
            validtxt_paths.append(valid_txt_path)
        else:
            print("can't find validtxt:", valid_txt_path)
    return validtxt_paths

def getEvaldataset_objlist(validtxt_paths):
    obj_list = []
    for eachvalidtxt in validtxt_paths:
#        break
        print("======loading {} evaldataset ======".format(Path(eachvalidtxt).parents[0].stem))
        evaldataset = EvalDataset(eachvalidtxt)
        print("len:", len(evaldataset.datasets))
        obj_list += evaldataset.datasets
    return obj_list


#%%

def main():
    date = '0529'
    cuda_num = 0
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

    process_num = 3
    eval_dir = os.path.join('./hmm_optimization/opt_history')
    
    ### assign the search range here: 
    ### note: it take a long time to finish if search space is too large
    temp_range = np.arange(60, 70, 10, dtype = int) 
    transition_lambda_range = np.arange(100, 160, 10, dtype = float) 
    observation_lambda_range = np.arange(16, 20, 8, dtype = int) 
    threshold_range = np.arange(0.40, 0.80, 0.05, dtype = float) 
    
    
    ### get paths of trained models ready for hmmopt
    ft_notes_dir = './hmm_optimization/hmmopt_todolist'
    ft_notes_csv = os.path.join(ft_notes_dir, 'hmmopt_todolist_0529.csv')
    
    df = pd.read_csv(ft_notes_csv, index_col = 0)
    
    target_model_dict = df.loc[df['hmm']=='ready']
    target_model_dict = target_model_dict.to_dict('records')
    
    
    ### get dataset for validation set
    main_dataset_dir = './datasets/original/'
    datasets_list = os.listdir(main_dataset_dir)
    validtxt_paths = getValidtxt_paths(datasets_list, main_dataset_dir, valid_fn = 'valid_audiofiles.txt')
    obj_list = getEvaldataset_objlist(validtxt_paths)
    
    song_num ='all'
    if song_num != 'all':
        obj_list = obj_list[0:song_num]
    else:
        song_num = len(obj_list)
    
    
    for modeldict in target_model_dict:
        # break

        exp_name = os.path.basename(modeldict['model_dir'])
        print("======Processing {} ======".format( exp_name ))
        
        ### save finetune information        
        evaluation_folder = eval_dir
        if not os.path.exists(evaluation_folder):
            Path(evaluation_folder).mkdir(parents = True, exist_ok = True)
        target_jsonpath = modeldict['model_dir']
        
        # if 'model_setting' in modeldict.keys():
        #     model_setting = modeldict['model_setting']
        ### model initialization
        if modeldict['model_type'] =='bsl_blstm':
            model = bsl_blstm()
        else:
            print("===!!!===> unknown model_type")
            
            
        ### finished training, not plot loss curve and apply hmm finetune
        model_fn = 'RNNBeatProc.pth'
        model_path = os.path.join(target_jsonpath , model_fn)
        ## load the best model just trained
        state = torch.load(model_path, map_location = device)
        model.load_state_dict(state)
        if torch.cuda.is_available():
            model.cuda(cuda_num)

        
        # csv save path
        csv_sdir = evaluation_folder
        evaluation_title = "HmmFT_"+str(song_num)+'songs_'+exp_name+ '_'
        best_para_spath = os.path.join(csv_sdir, evaluation_title+'bestParams_'+ date+'.csv')
        csv_spath = os.path.join(csv_sdir, evaluation_title+ date+'.csv')
    
        

        ### collect all activations and beat labels into list for multiprocess
        activation_list = []
        beatlabel_list = []
        for eachsong_obj in tqdm.tqdm(obj_list, desc = 'processing song:'):
#            break
            feat, beat, audiofile = eachsong_obj.get_data()
            activation = get_dlm_activation(model, device, feat)
            activation_list.append(activation)
            beatlabel_list.append(beat)
        
        args_list = []
        for ntempi in temp_range:
            for tlambda in transition_lambda_range:
                for olambda in observation_lambda_range:
                    for thres in threshold_range:
                        param_dict = {
                                'ntempi': ntempi, 
                                'tlambda': tlambda, 
                                'olambda': olambda, 
                                'thres': thres, }
                        args_list.append([param_dict, beatlabel_list, activation_list] )
        pool = Pool(process_num)
        results_dictlist = pool.map(process_worker, tqdm.tqdm(args_list))
                        
        csv_results = pd.DataFrame(results_dictlist)
        csv_results['model_dir'] = modeldict['model_dir']
        csv_results['model_simpname'] = modeldict['model_simpname']
        if 'model_setting' in modeldict.keys():
            csv_results['model_setting'] = modeldict['model_setting']
        else:
            csv_results['model_setting'] = ''
        csv_results['model_type'] = modeldict['model_type']
        best_parameters = csv_results.nlargest(5, "fuser_f sum")
        best_parameters.to_csv(best_para_spath)
        csv_results.to_csv(csv_spath)
            
if __name__ == "__main__":
    main()