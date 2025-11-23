
# DDU-Net: Learning Complex Vascular Topologies with  KAN-Swin Transformers and Double Dynamic Upsampler
This project leverages deep learning for cloud removal in full-disk solar images, encompassing both datasets, model parameters and network structure. 

**Title of the Article**: [DDU-Net: Learning Complex Vascular Topologies with  KAN-Swin Transformers and Double Dynamic Upsampler]

![image]([https://github.com/user-attachments/assets/df9370c6-4862-4654-b445-f7fa0be70be8](https://github.com/steve706/DDUM_final/blob/f97cb0ecb2f3fb61736dd28fb1a76a68ab821a90/image.png))
## Setup
Project Clone to Local
```
git clone https://github.com/steve706/DDUM_final.git
```
To install dependencies:

```
conda install python=3.7
conda install numpy=1.21.6
conda install pytorch=1.10
```
## Training
To train the models in the paper, run these commands:
```
python
--dataset=rose-svc
--init_lr=0.0007
--batch_size=3
--pn_size=3
--input_nc=3
--data_dir=./data/ROSE/ROSE-1/SVC
```
## Testing
To conduct testing, please download the model parameter zip without unzipping it and place it directly under a checkpoint folder. Set the input and output paths, run these commands:
```
python --dataset=rose-svc
--init_lr=0.0007
--batch_size=3
--pn_size=3
--input_nc=3
--data_dir=./data/ROSE/ROSE-1/SVC
--mode=test
--first_suffix=best_fusion.pth
```
