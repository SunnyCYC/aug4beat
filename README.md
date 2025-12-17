# aug4beat

This repo contains the source codes for paper titled 
"*Source Separation-based Data Augmentation for Improved Joint Beat and Downbeat Tracking*".
| [**Paper (arXiv)**](https://arxiv.org/abs/2106.08703) | [**Github**](https://github.com/SunnyCYC/aug4beat) |

In this work, we investigate a source separation-based approach for data augmentation for joint beat and downbeat tracking. Specifically, to account for the composition of the training data in terms of the percussive and non-percussive sound sources, we propose to employ a blind drum separation model to segregate the drum and non-drum sounds from each training audio signal, filtering out training signals that are drumless, and then use the obtained drum and non-drum stems to augment the training data. Experiment results validate the effectiveness of the proposed method, and accordingly the importance of drum sound composition in the training data for beat and downbeat tracking.



## Contents and usage of this repo
Note that some randomly selected songs of GTZAN dataset [1] are included in ./datasets/original/gtzan to illustrate the usage of this repo. As long as any other datasets are organized in same way, they can be processed by the provided scripts to generated drum/non-drum extra data for training.


The contents are oganized as follows:
* models: 
    * BaselineBLSTM.py : reproduced BLSTM-based model following [2]

* datasets:
    **Note: please make sure that each .wav has its correponding .beats, and the paired files are with same filename. Ex. blues.00000.wav -->blues.00000.beats**
    * original:
        * gtzan:
            * audio/
            * downbeats/
        * new_dataset(you may upload with same folder configuration):
            * audio/
            * downbeats/
    * sourcesep_aug (will be generated from original datasets by the scriptes provided below):
        * gtzan:
            * OnlyDrum/
            * NoDrum/
* usage of the scripts: please follow the below steps in order to generate drum/non-drum data and the adopted input features.
    * run traintest_split.py :
        * you may assign ratio of test set and valid set before run the script
        * the script will glob all wav paths of current datasets, and generate required .txt for the following procedures
    * run source_separation_aug4beat.py :
        * the script will conduct source separation using official 4-stem Spleeer [3, 4], and save the drum/non-drum stems in the desired directories.
        * .txt wil also be generated following same train/test/valid split of original data.
    * run OnlyDrum_selection.py :
        * this script will conduct the two drum selection methods (i.e. ABSM and OSFQ) mentioned in the paper. And the qualified stems will be saved.
    * run prepare_dataset.py :
        * this script will precompute and save the input features adopted in this paper
    * training:
        * train_bsl.py : can be used to train baseline models (non-augmented)
        * train_2combABSM.py : can be used to train models augmented by drum stems selected by ABSM
        * train_3combOSFQ.py : can be used to train models augmented by non-drum stems and drum stems selected by OSFQ.
        * users may modified the provided scripts to manipulate the composition of training data or to change the experiment settings. 
        * Note: please ensure the following code only collect folders of datasets.
         `mix_dataset_dirs = os.listdir(mix_main_dir) `
        Any unexpected folders (e.g. '.ipynb_checkpoints') could cause error, and should be excluded if generated. 
            * ex. add code like below to exclude it 
            `unwanted = ['.ipynb_checkpoints']`
            `mix_dataset_dirs = list(set(mix_dataset_dirs)-set(unwanted))`
    * HMM optimization:
        * After finishing training of several models, users may use the following scripts to find best parameters of HMM for each trained model.   
        * run organize_hmmopt_todolist.py : to generate a list of model containing the model directories for the optimization process.
        * run hmmopt.py : to find the best parameters and save for each model. Note that larger search space (i.e. temp_range, transition_lambda_range, etc.) could be assigned in the script to find best parameters for your models. Current space is reduced to save time.
        * run organize_hmmopt_bestparams.py: to organize best parameters of all models into a .csv file.
        * run Inference.py: read the .csv file and conduct beat/downbeat tracking using the models included.
        * **Note: since this repo only adopts 10 songs for training (as a demo), the model may not be trained well. A totally failed model could produce empty f-score during executoin of hmmopt.py and stop you from going further. Try upload more traning data or repeat traning a few times to solve this problem.**


    


---

## License

This project is licensed under the **MIT License** - see the [LICENSE](./LICENSE.txt) file for details.


## Reference
*[1] G. Tzanetakis and P. Cook, “Musical genre classification of audio signals,” IEEE Trans. Speech and Audio Processing, vol. 10, no. 5, pp. 293–302, 2002.*

*[2] S. B¨ock, F. Krebs, and G. Widmer, “Joint beat and downbeat tracking with recurrent neural networks,” Proc. Int. Soc. Music Inf. Retr. Conf., pp. 255–261, 2016.*

*[3] R. Hennequin, F. V. A. Khlif, and M. Moussallam, “Spleeter: A fast and state-of-the art music source separation tool with pre-trained models,” J. Open Source Softw., 2020.*

*[4] https://github.com/deezer/spleeter*
