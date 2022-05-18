# SPN2D-GAN in PyTorch
Paper: 《SPN2D-GAN: A Night-to-day Image-to-Image Translator Based on Semantic Prior Generator》

## Usage
```bash
├── datasets
   ├── ACDC
       ├── rgb_anon_trainvaltest
           └── ...
       └── gt_trainval
           └── ...
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.png
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.png
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.png
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.png
           ├── ddd.png
           └── ...
```
- The ACDC-night train and val set images need to be placed in `trainA` and `trainB`, and ACDC-night test set images need to be placed in `testA` and `testB`

### Prerequisites
- Linux Ubuntu + Anaconda
- Python 3.7
- PyTorch 1.5.0 + cuda 10.2 + cudnn 7.6.5 + torchvision 0.6.0
- dominate 2.6.0
- tensorboardX 2.2
- kornia 0.3.0

### Train
- Train a model:
```bash
> python train.py --dataroot ./datasets/YOUR_DATASET_NAME --name SPN2DGAN
```
- pretrained semantic mask extraction network and semantic supervision network model: [Google_drive](https://drive.google.com/drive/folders/1-bvVwYqlhm1zuU3W3GFLizzA2kKVufze?usp=sharing)
- Note that the path to the ACDC dataset needs to be changed manually in `data/unaligned_dataset.py`.


### Test
- Test the model:
```bash
> python test.py --dataroot ./datasets/YOUR_DATASET_NAME --name SPN2DGAN
```

### Metric
```bash
> python fid.py ./testResult/real_day ./testResult/fake_day --gpu 0
```
- The generated daytime images need to be placed in the `/fake_day` folder and the ACDC-test set real-world daytime images need to be placed in the `/real_day` folder.

## Dataset
Download the [ACDC-night](https://acdc.vision.ee.ethz.ch) dataset.

## Acknowledgments
Our code is inspired by [pytorch-CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
