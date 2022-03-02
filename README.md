# Behind the Curtain: Learning Occluded Shapes for 3D Object Detection (AAAI-2022)

## Reference

Please cite our paper if you are interested to use this implementation,  
 <strong>Behind the Curtain: Learning Occluded Shapes for 3D Object Detection</strong>.
```
@article{xu2021behind,
  title={Behind the Curtain: Learning Occluded Shapes for 3D Object Detection},
  author={Xu, Qiangeng and Zhong, Yiqi and Neumann, Ulrich},
  journal={arXiv preprint arXiv:2112.02205},
  year={2021}
}
```

The implementatin is also inspired by the ICCV-2021 paper,  
 <strong>SPG: Unsupervised domain adaptation for 3d object detection via semantic point generation</strong>.
``` 
@inproceedings{xu2021spg,
  title={Spg: Unsupervised domain adaptation for 3d object detection via semantic point generation},
  author={Xu, Qiangeng and Zhou, Yin and Wang, Weiyue and Qi, Charles R and Anguelov, Dragomir},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15446--15456},
  year={2021}
}
``` 


## Acknowledgement
Our model, BtcDet, is implemented based on [`[OpenPcdet 0.3.0]`](https://github.com/open-mmlab/OpenPCDet). We thank Shaohuai Shi for the discussion during our implementation.
  
  
     
## Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 16.04, should be able to work on 18.04)
* Python 3.6+
* PyTorch 1.1 or higher (tested on PyTorch 1.7, 1.8.1, 1.9, 1.10)
* CUDA 9.0 or higher (PyTorch 1.3+ needs CUDA 9.2+, tested on CUDA 10.2)
* [`spconv v1.2.1 (commit fad3000249d27ca918f2655ff73c41f39b0f3127)`](https://github.com/traveller59/spconv/commit/fad3000249d27ca918f2655ff73c41f39b0f3127)


### Install
a. Install the python environment manager `poetry` by following instructions on its [github page](https://github.com/python-poetry/poetry).

b. Run the following command to create the python environment:
```
make
```

c. Remember to add `poetry run` in front of `python` when you would like to run the python scripts.


## Preparation

### Use Our Preprocessed Data: 
you can use our generated kitti's data including the generated complete object points, download it [[here (about 31GBs)]](https://drive.google.com/drive/folders/1mK4akt3Qro9nbw_NRfP__p2nb3a_rzxv?usp=sharing)  and put the zip file inside data/kitti/ and unzip it as detection3d directory.

### Alternatively, Generate Data by Yourself:
####KITTI Dataset
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):
```
BtcDet
├── data
│   ├── kitti
    │   │   │──detection3d  │── ImageSets
                    │   │   │── training
                    │   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
                    │   │   │── testing
                    │   │   │   ├──calib & velodyne & image_2
```

* Generate the data infos by running the following command: 
```python 
poetry run python -m btcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
#### Generate Approximated complete object points:
(Under `BtcDet` directory, execute:)
```python 
poetry run python -m btcdet.datasets.multifindbestfit
```





## Run Training:
```
cd tools/
```
Single gpu training
```
mkdir output

mkdir output/kitti_car

poetry run python train.py --cfg_file ./cfgs/model_configs/btcdet_kitti_car.yaml --output_dir ../output/kitti_car/ --batch_size 2 --gpu_str "0"
```

Multi gpu training, assuming you have 4 gpus:
```
bash scripts/dist_train.sh 4  --batch_size 8 --gpu_str "0,1,2,3" --cfg_file ./cfgs/model_configs/btcdet_kitti_car.yaml --output_dir ../output/kitti_car/
```

## Run Testing:
```
cd tools/
```
Single gpu testing for all saved checkpoints, assuming you have 4 gpus:
```
poetry run python test.py --eval_all --cfg_file ./cfgs/model_configs/btcdet_kitti_car.yaml --gpu_str "0" --batch_size 2 --output_dir ../output/kitti_car/ --ckpt_dir  ../output/kitti_car/ckpt/
```

Multi gpu testing for all saved checkpoints, assuming you have 4 gpus:
```
bash scripts/dist_test.sh 4 --eval_all --cfg_file ./cfgs/model_configs/btcdet_kitti_car.yaml --gpu_str "0,1,2,3" --batch_size 8 --output_dir ../output/kitti_car/ --ckpt_dir  ../output/kitti_car/ckpt/
```

Multi gpu testing a specific checkpoint, assuming you have 4 gpus and checkpoint_39 is your best checkpoint :
```
bash scripts/dist_test.sh 4  --cfg_file ./cfgs/model_configs/btcdet_kitti_car.yaml --gpu_str "0,1,2,3" --batch_size 8 --output_dir ../output/kitti_car/ --ckpt  ../output/kitti_car/ckpt/checkpoint_epoch_39.pth
```
