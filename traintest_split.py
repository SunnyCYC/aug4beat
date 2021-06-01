# -*- coding: utf-8 -*-
"""
Created on Fri May 28 14:40:23 2021

@author: Sunny
"""

#%%
import numpy as np
import os
from sys import argv
from pathlib import Path
import glob

def TrainTestSplit(songlist, test_p = 0.1, valid_p = 0.1):
    song_num = len(songlist)
    test_n = int(song_num *test_p)
    valid_n = int(song_num *valid_p)
    idx = np.random.permutation(song_num)
    
    testlist = [songlist[i] for i in idx[:test_n]]
    validlist = [songlist[i] for i in idx[test_n:test_n+valid_n]]
    trainlist = [songlist[i] for i in idx[test_n+valid_n:]]
    
    return trainlist, testlist, validlist
    
def savePathtxt(spath, songlist):
    print("----> saving path txt :", spath)
    with open(spath, 'w') as file:
        for songpath in songlist:
            file.write(songpath+'\n')
            

def main():
    ori_data_dir = os.path.join('./', 'datasets', 'original')
    ori_datasets = glob.glob(os.path.join(ori_data_dir, '*'))
    current_path = os.getcwd()
    
    for each_dataset in ori_datasets:
        # break
        print("===> processing dataset:{} ===".format(each_dataset))
        audio_paths = [os.path.join(current_path, i) for i in glob.glob(os.path.join(each_dataset, "audio", "*.wav"))]
        
        ### save audio paths for future use
        audio_txtfn = os.path.join(each_dataset, "audio_files.txt")
        if not os.path.exists(audio_txtfn):
            savePathtxt(audio_txtfn, audio_paths)
    
        else:
            print("audio_files.txt exists")
        
        ### train, test, valid split
        trainlist, testlist, validlist = TrainTestSplit(audio_paths, test_p = 0.1, valid_p = 0.1)
        for ind, songlist in enumerate([trainlist, testlist, validlist]):
            spath = os.path.join(each_dataset, ['train_audiofiles.txt', 'test_audiofiles.txt', 'valid_audiofiles.txt'][ind])
            if not os.path.exists(spath):
                savePathtxt(spath, songlist)
            else:
                print("exists:", spath)
        
if __name__ == "__main__":
    main()