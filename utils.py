
import librosa
import shutil
import torch
import os
import numpy as np

import downbeat_dataset as mmdataset
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor)
from madmom.processors import ParallelProcessor, Processor, SequentialProcessor


global_sr = 44100


def getExMixset(exMix_dataset_dirs, folderName = 'features', abname = 'train_dataset.ab',
                drum_abname = "osfq_qualified.ab",
                main_dir = './datasets/sourcesep_aug/', 
                NoDrum= True, OnlyDrum = True):
    if NoDrum:
        nodrumset = mmdataset.load_dataset(os.path.join(main_dir, exMix_dataset_dirs[0],
                                            "NoDrum", folderName, abname))

        for dataset in exMix_dataset_dirs[1:]:
            ssaug_dir = os.path.join(main_dir, dataset)
            nodrumset+= mmdataset.load_dataset(os.path.join(ssaug_dir, "NoDrum", 
                                                            folderName, abname))
    if OnlyDrum:
        onlydrumset = mmdataset.load_dataset(os.path.join(main_dir, exMix_dataset_dirs[0], 
                                            "OnlyDrum", folderName, drum_abname))
        for dataset in exMix_dataset_dirs[1:]:
            ssaug_dir = os.path.join(main_dir, dataset)
            onlydrumset+= mmdataset.load_dataset(os.path.join(ssaug_dir, "OnlyDrum", 
                                                              folderName, drum_abname))
    if NoDrum and OnlyDrum:
        print("===using both NoDrum and OnlyDrum===")
        return nodrumset + onlydrumset
    elif NoDrum and not OnlyDrum:
        print("===using only NoDrum===")
        return nodrumset
    elif OnlyDrum and not NoDrum:
        print("===using only OnlyDrum===")
        return onlydrumset
    else:
        print("======Something is Wrong in Your getExMixset settings!!!======")

def getMixset(mix_dataset_dirs, folderName ='features', abname = 'train_dataset.ab',
              main_dir = './datasets/original/'):
    mixset = mmdataset.load_dataset(os.path.join(main_dir, mix_dataset_dirs[0], 
                                                 folderName, abname ))

    for dataset in mix_dataset_dirs[1:]:
        mixset_dir = os.path.join(main_dir, dataset, folderName)
        mixset += mmdataset.load_dataset(os.path.join(mixset_dir, abname ))
    return mixset



### calculating filtered spectrograms and first order derivative using Madmom API
def madmom_feature(wav):
    """ returns the madmom features mentioned in the paper"""
    sig = SignalProcessor(num_channels=1, sample_rate=global_sr )
    multi = ParallelProcessor([])
    frame_sizes = [1024, 2048, 4096]
    num_bands = [3, 6, 12]
    for frame_size, num_bands in zip(frame_sizes, num_bands):
        frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(
            num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
        spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
        diff = SpectrogramDifferenceProcessor(
            diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
        # process each frame size with spec and diff sequentially
        multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
    # stack the features and processes everything sequentially
    pre_processor = SequentialProcessor((sig, multi, np.hstack))
    feature = pre_processor.process( wav)
    return feature

### Functions for saving best models
### below functions were modified from source code: https://github.com/sigsep/open-unmix-pytorch/blob/master/openunmix/utils.py
def save_checkpoint(
    state, is_best, path, target):
    # save full checkpoint including optimizer
    torch.save(
        state,
        os.path.join(path, target + '.chkpnt')
    )
    if is_best:
        # save just the weights
        torch.save(
            state['state_dict'],
            os.path.join(path, target + '.pth')
        )


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, best_loss = None):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = best_loss
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta












