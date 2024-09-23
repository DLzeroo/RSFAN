# Region-guided Spatial Feature Aggregation Network for Vehicle Re-Identification

![framework14](https://github.com/user-attachments/assets/99550cfa-55ab-4b3b-adae-4979a1a1e3ec)

## Usage
### Requirements
We use single RTX A6000 48G GPU for training and evaluation. 
```
Python 3.8.18
pytorch 1.13.1
torchvision 0.14.1
torchaudio 0.13.1
timm 1.0.8
scipy 1.10.1
yacs 0.1.6
```
### Dataset Preparation
Download the VeRi-776 dataset from [here](https://github.com/JDAI-CV/VeRidataset), VehicleID dataset from [here](https://pkuml.org/resources/pku-vehicleid.html) and VERI-Wild dataset form [here](https://github.com/PKU-IMRE/VERI-Wild).

Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <VeRi>/
|       |-- image_query
|       |-- image_test
|       |-- image_train
|       |-- ...
|
|   |-- <VehicleID_V1.0>/
|       |-- attribute
|       |-- image
|       |-- train_test_split
|            |-- test_list_800.txt
|            |-- test_list_1600.txt
|            |-- ...
|
|   |-- <VERI-Wild>/
|       |-- gallery
|       |-- images
|       |-- query
|       |-- ...
```
## Training

```python
python -W ignore tools/train.py --config_file='configs/veri.yml'
```

## Testing

```python
python -W ignore tools/test.py --config_file='configs/veri.yml'
```
## Results
Download the video of test on the VeRi-776 dataset from [here](https://pan.baidu.com/s/1AubvbI96R4H5Z-azhZKAVg?pwd=v42r).
Download the trained weights on the VeRi-776 dataset from [here](https://pan.baidu.com/s/1c7i9brp99YBykdp2ryRVFA?pwd=2e5c).

## Citation
If you find this code useful for your research, please cite our paper.

```tex
@article{RSFAN,
   title={Region-guided Spatial Feature Aggregation Network for Vehicle Re-Identification}, 
   author={Yanzhen Xiong and Jinjia Peng and Zeze Tao and Huibing Wang},
   journal={Engineering Applications of Artificial Intelligence}, 
   year={2024},
   volume={},
   number={},
   pages={},
   doi={}
}
```
