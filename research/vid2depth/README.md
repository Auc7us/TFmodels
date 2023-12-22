![TensorFlow Requirement: 1.15](https://img.shields.io/badge/TensorFlow%20Requirement-1.15-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# Meta Learning on vid2depth

**vid2depth = Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints**

Reza Mahjourian, Martin Wicke, Anelia Angelova

CVPR 2018

Project website: [https://sites.google.com/view/vid2depth](https://sites.google.com/view/vid2depth)

ArXiv: [https://arxiv.org/pdf/1802.05522.pdf](https://arxiv.org/pdf/1802.05522.pdf)

<p align="center">
<a href="https://sites.google.com/view/vid2depth"><img src='https://storage.googleapis.com/vid2depth/media/sample_video_small.gif'></a>
</p>

<p align="center">
<a href="https://sites.google.com/view/vid2depth"><img src='https://storage.googleapis.com/vid2depth/media/approach.png' width=400></a>
</p>


## 1. Installation

### Requirements

#### Python Packages

```shell
conda create -n venv python=3.7 # Recommended: create a virtual environment.
conda install numpy=1.18.5
conda install matplotlib
conda install tensorflow-gpu=1.15
```

### Download vid2depth

```shell
https://github.com/Auc7us/TFmodels.git
```

## 2. Datasets

### Download KITTI dataset (174GB) (optional)

```shell
mkdir -p ~/vid2depth/kitti-raw-uncompressed
cd ~/vid2depth/kitti-raw-uncompressed
wget https://raw.githubusercontent.com/mrharicot/monodepth/master/utils/kitti_archives_to_download.txt
wget -i kitti_archives_to_download.txt
unzip "*.zip"
```

### Download Cityscapes dataset (110GB) (optional)

You will need to register in order to download the data.  Download the following
files:

* leftImg8bit_sequence_trainvaltest.zip
* camera_trainvaltest.zip

### Download Bike dataset (34GB)

Please see [https://research.google/tools/datasets/bike-video/](https://research.google/tools/datasets/bike-video/)
for info on the bike video dataset.

Special thanks to [Guangming Wang](https://guangmingw.github.io/) for helping us
restore this dataset after it was accidentally deleted.

```shell
mkdir -p ~/vid2depth/bike-uncompressed
cd ~/vid2depth/bike-uncompressed
wget https://storage.googleapis.com/vid2depth/dataset/BikeVideoDataset.tar
tar xvf BikeVideoDataset.tar
```

## 3. Inference

### Download trained model

```shell
mkdir -p ~/vid2depth/
cd ~/vid2depth/
git clone https://github.com/Auc7us/vid2depth-ckpts.git
mv vid2depth-ckpts trained-model-Bike
cd trained-model-Bike
rm model-185802.data-00000-of-00001
wget https://github.com/Auc7us/vid2depth-ckpts/raw/master/model-185802.data-00000-of-00001?download=
```

### Run inference

```shell
cd tensorflow/models/research/vid2depth
python inference.py \
  --dataset_dir ~/vid2depth/bike-uncompressed \
  --output_dir ~/vid2depth/inference \
  --dataset_video 2806 \
  --model_ckpt ~/vid2depth/trained-model-Bike/model-185802
```

## 4. Training

### Prepare KITTI training sequences

```shell
# Prepare training sequences.
cd ~/TFmodels/research/vid2depth
python dataset/gen_data.py \
  --dataset_name kitti_raw_eigen \
  --dataset_dir ~/vid2depth/kitti-raw-uncompressed \
  --data_dir ~/vid2depth/data/kitti_raw_eigen \
  --seq_length 3
```

### Prepare Cityscapes training sequences (optional)

```shell
# Prepare training sequences.
cd ~/TFmodels/research/vid2depth
python dataset/gen_data.py \
  --dataset_name cityscapes \
  --dataset_dir ~/vid2depth/cityscapes-uncompressed \
  --data_dir ~/vid2depth/data/cityscapes \
  --seq_length 3
```

### Prepare Bike training sequences (optional)

```shell
# Prepare training sequences.
cd ~/TFmodels/research/vid2depth
python dataset/gen_data.py \
  --dataset_name bike \
  --dataset_dir ~/vid2depth/bike-uncompressed \
  --data_dir ~/vid2depth/data/bike \
  --seq_length 3
```

### Compile the ICP op

The pre-trained model is trained without using the ICP loss.  It is possible to run
inference on this pre-trained model with compiling the ICP op.  It is also
possible to train a new model from scratch without compiling the ICP op by
setting the icp loss to zero.

If you would like to compile the op and run a new training job using it, please
use the CMakeLists.txt file at
[https://github.com/IAMAl/vid2depth_tf2/tree/master/ops](https://github.com/IAMAl/vid2depth_tf2/tree/master/ops).

### Run training

```shell
# Train
cd tensorflow/models/research/vid2depth
python train.py \
  --data_dir ~/vid2depth/data/bike \
  --seq_length 3 \
  --reconstr_weight 0.85 \
  --smooth_weight 0.05 \
  --ssim_weight 0.15 \
  --icp_weight 0 \
  --checkpoint_dir ~/vid2depth/checkpoints
```

### Run MAML Training
```shell
python MAML.py \
  --data_dir /home/flash/vid2depth/data/bike \
  --num_tasks 5 \
  --num_inner_updates 2 \
  --inner_lr 0.0002 \
  --meta_lr 0.001 \
  --num_epochs 4
```
