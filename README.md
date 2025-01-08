# Monodepth - Single image depth estimation
This is Pytorch implementation for training and evaluating method described in

> **Unsupervised Monocular Depth Estimation with Left-Right Consistency**
>
> [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)
>
> [CVPR 2017 (arXiv pdf)](https://arxiv.org/pdf/1609.03677v3)

## ⚙️ Setup
Install [Pytorch](https://pytorch.org/) and requirements in `requirements.txt` file.
```shell
pip install -r requirements.txt
```

## 💾 KITTI training data
You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
sudo apt-get install parallel > /dev/null
bash data/parallel-data-download-script.sh
```
The script above would download and extract data into folders.
