# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 20:05:13 2020

@author: CITI
"""


#%%
""" get normal train audio files , find the corresponding OnlyDrum, 
check the abs_mean values, save the qualified files into list """
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import tqdm
import librosa
from multiprocessing import Pool
import pickle
import pandas as pd
import downbeat_dataset as mmdataset

target_run_dir = './datasets/sourcesep_aug/'

def get_songpaths(train_audio_files_normal_path):
    train_songpaths = []
    with open(train_audio_files_normal_path) as f:
        for line in f.readlines():
            train_songpaths.append(line.strip('\n'))
    return train_songpaths 

def get_onsetFQ_absMean(train_songpaths):
    onlydrum_osfq_absmean  = []
    for eachsong in tqdm.tqdm(train_songpaths):

        onlydrumpath = eachsong
        if onlydrumpath.endswith('.flac'):
            onlydrumpath = onlydrumpath.replace('.flac', '.wav')
        if not os.path.exists(onlydrumpath):
            print("Can't find this song:", onlydrumpath)
        audio, rate = sf.read(onlydrumpath)
        absmean = abs(audio).mean()
        o_env = librosa.onset.onset_strength(audio, sr=rate)
        onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=rate)
        onsetFQ = len(onset_frames)/(len(audio)/rate)
        onlydrum_osfq_absmean.append([onlydrumpath, absmean, onsetFQ])
    return onlydrum_osfq_absmean            

def absmean_worker(each_targetdataset):
    train_audio_files_normal_path = os.path.join(target_run_dir, each_targetdataset,'OnlyDrum', 'train_audiofiles.txt')
    train_songpaths = get_songpaths(train_audio_files_normal_path)
    try:
        onlydrum_osfq_absmean = get_onsetFQ_absMean(train_songpaths)
    except:
        print("error when getting absmean for :", each_targetdataset)
            
    return {each_targetdataset: onlydrum_osfq_absmean}

def get_qualifiedSongs(absmeanlist, absmean_threshold = 0.001, OnsetFQ_threshold = 1.0):
    qualified_list = []
    for songpath, absmean, onsetFQ in absmeanlist:
        if absmean >= absmean_threshold and onsetFQ> OnsetFQ_threshold:
            qualified_list.append([songpath, absmean, onsetFQ])
    return qualified_list

def grepSongs(dataset_name, absmeanlist):
    song_list = []
    for eachsongpath, absmean, osfq in absmeanlist:
        song_datasetname = Path(eachsongpath).parents[1].stem
        if song_datasetname == dataset_name:
            song_list.append(eachsongpath)
        else:
            pass
    return song_list

def groupSongList(absmeanlist):
    """ input: absmeanlist with songpathlist and absmean, 
    output: dictionary with datasetdirs as key, songlists as values """
    
    datasets_list = []
    grouped_dict = {}
    for eachsongpath, absmean, osfq in absmeanlist:
#        break
        dataset_name = Path(eachsongpath).parents[1].stem        
        if dataset_name not in datasets_list:
            datasets_list.append(dataset_name)
            song_list = grepSongs(dataset_name, absmeanlist)
            grouped_dict[dataset_name] = song_list
    return grouped_dict
    
def makeQualifab(pk_qualified_fn, ab_sname = 'osfq_qualified.ab', dataset_main_dir = './datasets/sourcesep_aug'):
    with open(pk_qualified_fn, 'rb') as file:
        qualified_onlydrum_dlist = pickle.load(file)
    #main_exp_dir = '/home/sunnycyc/NAS_189/home/BeatTracking/LeaveOneOut_experiments/run4_fps_100_SSAUG_normal/'

    main_exp_dir = dataset_main_dir
    for exp_folder, absmeanlist, numQsongs in tqdm.tqdm(qualified_onlydrum_dlist):
        # break
        grouped_dict = groupSongList(absmeanlist)
        exp_folder_path = os.path.join(main_exp_dir, exp_folder, "OnlyDrum", "features")
        onlydrum_trainset_normal_ab_path = os.path.join(exp_folder_path, ab_sname)
        
        if not os.path.exists(onlydrum_trainset_normal_ab_path):
            if not grouped_dict:
                ### if no qualified song for that dataset, create empty .ab file
                print("found empty dict for :", exp_folder)
                columns = ['audio_file','beats_file', 'feature_file', 
               'offset', 'duration', 'length', 'song_duration', 'name'] 
                df = pd.DataFrame(columns=columns)
                df.to_csv(onlydrum_trainset_normal_ab_path)
            else:
                for ind, (dataset_name, songlist) in enumerate(grouped_dict.items()):
            #        break
                    dataset_dir = os.path.join(dataset_main_dir, dataset_name, 'OnlyDrum' )
                    onlydrum_trainset_normal_save_dir = os.path.join(dataset_dir, 'cutfeatures')
            
                    if ind ==0:
                        onlydrum_trainset_normal = mmdataset.AudioBeatsDatasetFromList(
                                                    songlist, onlydrum_trainset_normal_save_dir, 
                                                    force_nb_samples = None, audio_list = True, 
                                                    dataset_path = dataset_dir)
                    else:
                        onlydrum_trainset_normal += mmdataset.AudioBeatsDatasetFromList(
                                                    songlist, onlydrum_trainset_normal_save_dir, 
                                                    force_nb_samples = None, audio_list = True, 
                                                    dataset_path = dataset_dir)
                onlydrum_trainset_normal.save(onlydrum_trainset_normal_ab_path)
        else:
            print("ab exists:", onlydrum_trainset_normal_ab_path)
#%%
def main():
    
    target_datasets = os.listdir(target_run_dir)
    process_num = 3
    pool = Pool(process_num)
    results_dictlist = pool.map(absmean_worker, tqdm.tqdm(target_datasets))
    selection_dir = './drumselection/'
    
    if not os.path.exists(selection_dir):
        Path(selection_dir).mkdir(parents = True, exist_ok = True)
        
    pk_spath = os.path.join(selection_dir, 
                            'OSFQ_absmean.pickle')
    with open(pk_spath, 'wb') as file:
        pickle.dump(results_dictlist, file)
    
    
    ### read the saved osfq, absm and perform selection
    with open(pk_spath, 'rb') as file:
        abs_mean_onlydrum4train = pickle.load(file)
    
    absm_qualified_songs_perdataset = []
    osfq_qualified_songs_perdataset = []
    for eachdict in abs_mean_onlydrum4train:
    #    break
        dataset_name = list(eachdict.keys())[0]
        absmeanlist = eachdict[dataset_name]
        osfq_qualified_list = get_qualifiedSongs(absmeanlist, absmean_threshold = 0.001, OnsetFQ_threshold = 1.0)
        absm_qualified_list = get_qualifiedSongs(absmeanlist, absmean_threshold = 0.01, OnsetFQ_threshold = -0.1)
        osfq_qualified_songs_perdataset.append([dataset_name, osfq_qualified_list, len(osfq_qualified_list)])
        absm_qualified_songs_perdataset.append([dataset_name, absm_qualified_list, len(absm_qualified_list)])
        print("{} dataset has {} absm qualified songs and {} osfq qualified songs ".format(dataset_name, 
                                                                                           len(absm_qualified_list), 
                                                                                           len(osfq_qualified_list)))
    
    pk_absm_qualified_fn = os.path.join(selection_dir, 
                            'absm_qualified.pickle')
    pk_osfq_qualified_fn = os.path.join(selection_dir, 
                            'osfq_qualified.pickle')
    with open(pk_absm_qualified_fn, 'wb') as file:
        pickle.dump(absm_qualified_songs_perdataset, file)
    with open(pk_osfq_qualified_fn, 'wb') as file:
        pickle.dump(osfq_qualified_songs_perdataset, file)
        
    for pk_qualified_fn in [pk_absm_qualified_fn, pk_osfq_qualified_fn]:
        # break
        makeQualifab(pk_qualified_fn, ab_sname = Path(pk_qualified_fn).stem+'.ab', dataset_main_dir = './datasets/sourcesep_aug')

if __name__ =="__main__":
    main()