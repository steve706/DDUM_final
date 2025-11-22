
# DDU-Net: Learning Complex Vascular Topologies with  KAN-Swin Transformers and Double Dynamic Upsampler
This project leverages deep learning for cloud removal in full-disk solar images, encompassing both datasets, model parameters and network structure. 

**Title of the Article**: [DDU-Net: Learning Complex Vascular Topologies with  KAN-Swin Transformers and Double Dynamic Upsampler]

![5](https://github.com/user-attachments/assets/df9370c6-4862-4654-b445-f7fa0be70be8)
## Setup
Project Clone to Local
```
git clone https://github.com/dupeng24/full-disk-cloud-removal.git
```
To install dependencies:

```
conda install python=3.11.4
conda install pytorch-cuda=11.8
conda install numpy=1.25.0
conda install scikit-image=0.20.0
conda install h5py=3.9.0
```
## Cloud Detection and Classification
To conduct cloud detection and categorization, please change the name of the corresponding folder first, run these commands:
```
python finallyquality.py
```
## Training
Please download the data compression package full-disk images data.zip, divide the training set in it into cloud and clean, please make sure that the cloudy image and the labeled image have the same name. Please run the following to generate the h5 dataset file:
```
python makedataset.py
```
To train the models in the paper, run these commands:
```
python train.py
```
## Testing
To conduct testing, please download the model parameter zip without unzipping it and place it directly under a checkpoint folder. Set the input and output paths, run these commands:
```
python test.py
```
