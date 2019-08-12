# Watch-Attend-and-Parse-tensorflow-version
This is a fork of [wwjwhen's tensorflow implementation](https://github.com/wwjwhen/Watch-Attend-and-Parse-tensorflow-version) of the [JianshuZhang's WAP system](https://github.com/JianshuZhang/WAP/).\
The fork is more or less a homebrew version which implements the paper's multiscale attention model.\
The original README.md is in [./README_OG.md](./README_OG.md).

# Requirements
The original requirements were:
  - Tensorflow v1.6
  - Python 3
  - cuda 9.0 [optional]
  - cudnn 7 [optional]
  
This folk was tested and run on the conda enviroments specified in [./environment.yml](./environment.yml) 

# Data preparation
- Vocabulary file:  A text file with format: character \t index \n
- Training and validating feature files: Pickle files each contains a python dict with keys are the images' ids (normally the images' paths) pointing to 3D numpy matrice [C, H, W]
- Label files: Text files with format: id \t label seperated by space \n

There are example files including data and data generating scripts under [./data](./data)

The generating scrips were written in python2\
If your new pkl file is written in python3, you may want to edit out the latin1 encoding option in every pickle.load call.
# Files layout
- [./model-distributed.py](./model-distributed.py): Multi GPU version from wwjwhen's repo. Untouched, use at your own risk.
- [./model-single-GPU.py](./model-single-GPU.py): Densenet single-scale version
- [./model-single-GPU-multi-scale.py](./model-single-GPU-multi-scale.py): Densenet 2-scale attention version
- [./model-single-GPU-multi-scale-ABC.py](./model-single-GPU-multi-scale-ABC.py): Densenet 3-scale attention version
- *-decode.py: Ripped down version for predict only.

# Usage
```
python <files> <parameters>
```
_(duh)_

# Parematers
- dictPath: path to vocabulary.txt file
- trainPklPath: path to trainning .pkl file
- trainCaptionPath: path to training .txt label file
- validPklPath
- validCaptionPath
- resultPath: directory to store predictions and metric results (WER and acc)
- --resultFileName: result file name, will automatically be appended .txt and .wer in validation process
- --logPath: file path to store logging info, should include .txt
- --savePath: directory to save model every epoch
- --saveName: name of model to be saved

# Notes:
With multiscale version, modify the encoder branching model by editing the B_option (and C_option) variable in the main function
_e.g:_
_`B_option = {"branch_from": 1, "growth_rate": 24, "level": 8}`_

__Warning__: The saving current saving mechanism is simply overwriting the previous save at each epoch. Best model saving is not yet to be implemented.

__Warning__:
With multiscale version, when decoding, be sure to double check the B_option, C_option variable to match with those in the training phase.