
# DDU-Net: Learning Complex Vascular Topologies with  KAN-Swin Transformers and Double Dynamic Upsampler
This project employs deep learning techniques to segment complex vascular topologies in OCTA images, integrating comprehensive considerations of datasets, and network architecture.

We propose the KAN-Swin Transformer, an encoder block that replaces this rigid component with an adaptive operator based on Kolmogorov-Arnold Networks (KANs). This new layer features B-spline-based learnable activation functions on network edges, rather than fixed functions on nodes, empowering the encoder to learn geometrically-aware representations specific to intricate morphologies like bifurcations and high-tortuosity segments. The decoder features a novel dual-path Double Dynamic Upsampler Module (DDUM), which processes edge-rich shallow features and semantic deep features in parallel before an attention-based fusion, avoiding feature contamination. An Information Compensation Module (ICM) further recovers fine details using multi-dilation convolutions. For challenging low-contrast Inner Vascular Complex (IVC) images, we introduce a multimodal fusion strategy, where a Feature Alignment Module (FAM) aligns probability maps from auxiliary modalities to enhance the IVC representation.



**Title of the Article**: [DDU-Net: Learning Complex Vascular Topologies with  KAN-Swin Transformers and Double Dynamic Upsampler]
![image](https://github.com/steve706/DDUM_final/blob/main/visualize_1.png)

## 🏆 Key Innovations

### 1\. KAN-Swin Transformer (The Core Innovation)

Unlike traditional Transformers that rely on MLPs with fixed activations (e.g., GELU), our **KAN-Swin Transformer** utilizes **Kolmogorov-Arnold Networks**.

  * **Learnable Activations:** Places learnable B-spline activation functions on edges.
  * **Adaptive Topology:** The network *learns* the optimal non-linear function required to model complex vessel curvatures.

### 2\. Double Dynamic Upsampler (DDUM)

  * **Disentangled Decoding:** Splits decoding into a "Shallow Path" (for high-freq details) and a "Deep Path" (for semantic integrity).
  * **Attention Fusion:** Dynamically fuses these paths only at the final stage to prevent feature contamination.

### 3\. Multimodal Fusion with FAM

  * **Clinical Workflow:** Designed specifically for low-contrast **IVC** images.
  * **Alignment:** Uses deformable convolutions to align probability maps from SVC and DVC layers before fusing them with the IVC input.

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
## 📝 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{shang2024ddunet,
  title={DDU-Net: Learning Complex Vascular Topologies with KAN-Swin Transformers and Double Dynamic Upsampler},
  author={Shang, Zhenhong and Li, Jun},
  journal={Physics in Medicine & Biology (Under Review)},
  year={2025}
}
## 🤝 Acknowledgements

We thank the authors of the **ROSE-1**, **Prevent**, and **OCTA500** datasets for making their data publicly available. We also acknowledge the official implementation of **Efficient-KAN** which inspired our encoder design.
