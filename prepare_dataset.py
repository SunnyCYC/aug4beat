# -*- coding: utf-8 -*-




import os
import numpy as  np
import downbeat_dataset as mmdataset
import glob



def make_ab(audio_files, dataset_dir, ab_name = "train_dataset.ab"):
    where_to_save = os.path.join(dataset_dir, 'features')
    dataset = mmdataset.AudioBeatsDatasetFromList(
                audio_files, where_to_save, force_nb_samples=None)
    ab_spath = os.path.join(where_to_save, ab_name)
    dataset.save(ab_spath)
    return ab_spath

def make_dataset(dataset_dir):

    # Make a training/validation dataset
    train_audio_files   = os.path.join(dataset_dir, 'train_audiofiles.txt')
    valid_audio_files = os.path.join(dataset_dir, 'valid_audiofiles.txt')
    ## for evaluation, we use fulllength features, 
    ## therefore test_audio_files are not required here
    # test_audio_files = os.path.join(dataset_dir, 'test_audiofiles.txt')
    
    train_ab = make_ab(train_audio_files, dataset_dir, ab_name = "train_dataset.ab")
    valid_ab = make_ab(valid_audio_files, dataset_dir, ab_name = "valid_dataset.ab")
    return train_ab, valid_ab

def main():
    ### get all dataset folders
    dataset_dir_list = glob.glob(os.path.join('./', 'datasets', "*"))
    datasets = []
    
    for dataset_dir in dataset_dir_list:
        if not '/sourcesep_aug' in dataset_dir:
            datasets += glob.glob( os.path.join(dataset_dir, "*"))
        else:
            aug_datasets = glob.glob(os.path.join(dataset_dir, "*"))
            for aug_dataset in aug_datasets:
                aug_folders = [i for i in  glob.glob(os.path.join(aug_dataset, "*")) if i.endswith('NoDrum') or i.endswith('OnlyDrum')]
                datasets += aug_folders 
                
    ### save the dataset information in csv-like .ab files
    for dataset_dir in datasets:
        train_ab, valid_ab = make_dataset(dataset_dir)
        #### load and precompute features
        trainset = mmdataset.load_dataset(train_ab)
        validset = mmdataset.load_dataset(valid_ab)
        
        trainset.precompute()
        validset.precompute()

    
#%%
if __name__ =='__main__':
    main()
    