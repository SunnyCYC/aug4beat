#%%
import os 
import glob 
import subprocess
from pathlib import Path

from pydub import AudioSegment
import tqdm
import shutil

target_sr = 44100


def readSpath(txtpath):
    songpath_list = []
    with open(txtpath, 'r') as file:
        for line in file.readlines():
            songpath_list.append(line.strip('\n'))
    return songpath_list

def mergingstems(stems4merge, NoDrum_spath):
    NoDrum_audio = AudioSegment.from_file(stems4merge[0]).set_frame_rate(target_sr).set_channels(1)
    for eachstem in stems4merge[1:]:
        new_stem = AudioSegment.from_file(eachstem).set_frame_rate(target_sr).set_channels(1)
        NoDrum_audio = NoDrum_audio.overlay(new_stem)
    NoDrum_audio.export(NoDrum_spath, format = 'wav')

def get_datasetSongs_NmergeNsave(datasetName , 
                     SourceSepAUG_dataset_dir ):
    songdir_list = glob.glob(os.path.join(SourceSepAUG_dataset_dir, datasetName, 'SpleeterSep_temp', "*"))
    for eachsong_path in tqdm.tqdm(songdir_list):
#        break
        #### glob all stems
        spleeter_wavs = glob.glob(os.path.join(eachsong_path, "*.wav"))
        #### separate pathlist for drum and nodrum
        drumstem_src = os.path.join(eachsong_path, "drums.wav")
        stems4merge = list(set(spleeter_wavs)-set([drumstem_src]))
        NoDrum_spath = eachsong_path.replace('SpleeterSep_temp', 'NoDrum')+".wav"
        if not os.path.exists(NoDrum_spath):
            mergingstems(stems4merge, NoDrum_spath)
    
        drumwav_spath = eachsong_path.replace('SpleeterSep_temp', 'OnlyDrum')+".wav"
        if not os.path.exists(drumwav_spath):
            drumaudio = AudioSegment.from_file(drumstem_src).set_frame_rate(target_sr).set_channels(1)
            drumaudio.export(drumwav_spath, format = 'wav')

def getAugpath(ori_songtxt, datasetname, augtype = 'NoDrum'):
    songlist = []
    with open(ori_songtxt, 'r') as file:
        for line in file.readlines():
            # break
            songlist.append(line.replace('/datasets/original', '/datasets/sourcesep_aug').replace(datasetname+'/audio', datasetname+'/'+augtype))
    return songlist

def savetxt(spath, songlist):
    with open(spath, 'w') as file:
        for song in songlist:
            file.write(song)
    
            
def main():
    """ 
    glob all datasets in original, create audiofile.txt and perform source separation on all songs
    """
    ori_dir = os.path.join('./', 'datasets/original/')
    ori_datasets = glob.glob(os.path.join( ori_dir, "*"))
    ssaug_dir = os.path.join('./', 'datasets/sourcesep_aug/')
    fail_list = []

    ### applying source separation on eachsong in each dataset
    for eachdataset in ori_datasets:
        audio_file_path = os.path.join(eachdataset, "audio_files.txt")
        ##### get all songs for the dataset #####
        song_path_list = readSpath(audio_file_path)

        ##### applying spleeter on each song #####
        for eachsong in song_path_list:
            out_dirname = os.path.join(ssaug_dir, os.path.basename(eachdataset),
                                       "SpleeterSep_temp")
            if not os.path.exists(out_dirname) :
                print("create folder:", out_dirname)
                Path(out_dirname).mkdir(parents = True, exist_ok = True)
            print("Processing dataset:{}, song:{} ".format( eachdataset, Path(eachsong).stem))
            try:
                p = subprocess.Popen(["spleeter", "separate", "-i", str(eachsong), "-p", "spleeter:4stems", "-o", out_dirname ])
                p.communicate()
            except:
                fail_list.append([eachdataset, eachsong])
                
    errors_n = len(fail_list)
    print("======> finished processing with {} error songs========".format(errors_n))
    if errors_n > 0:
        print("failed songs:", fail_list)

    ### merging/organizing nondrum/drum stems for each dataset
    print("========> Merging Spleeter Stems ========")
    SourceSepAUG_dataset_list = os.listdir(ssaug_dir)

    for eachdataset in SourceSepAUG_dataset_list:
        NoDrum_dir = os.path.join(ssaug_dir, eachdataset, "NoDrum")
        if not os.path.exists(NoDrum_dir):
            Path(NoDrum_dir).mkdir(parents = True, exist_ok= True)
        OnlyDrum_dir = os.path.join(ssaug_dir, eachdataset, "OnlyDrum")
        if not os.path.exists(OnlyDrum_dir):
            Path(OnlyDrum_dir).mkdir(parents = True, exist_ok = True)
        print("======= Processing {} Dataset =======".format(eachdataset))
        get_datasetSongs_NmergeNsave(eachdataset , 
                         SourceSepAUG_dataset_dir = ssaug_dir)
    
        #### copy train-test-valid files to nodrum/drum folder
        #### copy downbeat annotation folders to aug folders
        downbeat_src = os.path.join(ori_dir, eachdataset, 'downbeats')
        print("=======> copying traintest split to aug folders...")
        oritxts = [os.path.join(ori_dir, eachdataset , i) for i in ['audio_files.txt', 'train_audiofiles.txt', 
                                                       'test_audiofiles.txt', 'valid_audiofiles.txt']]
        for oritxt in oritxts:
            # break
            for augtype in ["NoDrum", "OnlyDrum"]:
                # break
                augsonglist = getAugpath(oritxt, eachdataset, augtype)
                augspath = os.path.join(ssaug_dir, eachdataset, augtype, os.path.basename(oritxt))
                if not os.path.exists(augspath):
                    savetxt(augspath, augsonglist)
        
                downbeat_dst = os.path.join(ssaug_dir, eachdataset, augtype, 'downbeats')
                if not os.path.exists(downbeat_dst):
                    print("----> copying annotations to: ", downbeat_dst)
                    shutil.copytree(downbeat_src, downbeat_dst)

if __name__ =="__main__":
    main()