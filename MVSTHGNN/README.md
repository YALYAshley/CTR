## Introduction
The official implementation of **Skeleton Temporal Hypergraph Neural Networks for Person Re-Identification**.

## Installation
We use python=3.7 and pytorch=1.1.0.

## Dataset preparation
Download datasets to `data/` in the project root directory.

**iLIDS-VID**

Download dataset from [iLIDS-VID](http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html).
The code also supports automatic download for iLIDS-VID. Simple use `-d ilidsvid` when running the training code.

**PRID-2011**

Download dataset from [PRID-2011](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/).

**MARS**

Download dataset from [Baidu Yun](http://pan.baidu.com/s/1hswMDfu) or [Google Driver](https://drive.google.com/folderview?id=0B6tjyrV1YrHeMVV2UFFXQld6X1E&usp=sharing).

Organize the data structure as:
```shell script
©À©¤©¤ ilids-vid
©¦   ©À©¤©¤ i-LIDS-VID
©¦   ©À©¤©¤ pose.json
©¦   ©À©¤©¤ splits.json
©¦   ©¸©¤©¤ train-test people splits
©¦
©À©¤©¤ mars
©¦   ©À©¤©¤ bbox_test
©¦   ©À©¤©¤ bbox_train
©¦   ©À©¤©¤ info
©¦   ©À©¤©¤ pose.json
©¦
©À©¤©¤ prid2011
    ©À©¤©¤ pose.json
    ©À©¤©¤ prid_2011
    ©À©¤©¤ splits_prid2011.json
```
`pose.json` for the three datasets are obtained by [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose), we also 
provide those files on [Baidu Yun](https://pan.baidu.com/s/1bKLYO5eiPUmdEzeEPV7xMQ) (code: txk1) 
or [Google Driver]().  

## Train
```shell script
git clone this_project
cd ST_HGNN_reid/
```
For training on iLIDS-VID:
```shell script
bash scripts/ilidsvid.sh
```
For training on PRID-2011:
```shell script
bash scripts/prid2011.sh
```
For training on MARS:
```shell script
bash scripts/mars.sh
```

To use multiple GPUs, you can use `--gpu-devices 0,1,2,3` in the training script.

## Citation
Please kindly cite this project in your paper if it is helpful?:
```
(unfinished)

```


## Acknowledgements
The project is developed based on [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) and [HGNN](https://github.com/iMoonLab/HGNN).