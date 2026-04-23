# [Paper Title: Unsupervised 3D Human Pose Estimation via Conditional Multi-view Ancestral Sampling]

This repository is the official implementation of the paper **"[Unsupervised 3D Human Pose Estimation via Conditional Multi-view Ancestral Sampling]"**, which has been accepted to **FG 2026**.

## News
- **[April 2026]**: Paper accepted to FG 2026! 
- **[Coming Soon]**: We are currently cleaning up the code.

## Getting started

### 1. Setup Environment
To set up the environment, we recommend using Conda. Run the following commands to create a dedicated environment and install the required dependencies:

```shell
# Create a new conda environment
conda create -n cmas python=3.9
conda activate cmas

# Install dependencies
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

*(Note: Please ensure you have the appropriate CUDA version installed for PyTorch.)*

### 2. Get Data
To evaluate the model, download the **3DYoga90** dataset. 

* **3DYoga90 Dataset**: [https://github.com/seonokkim/3DYoga90](https://github.com/seonokkim/3DYoga90)

The 3DYoga90 dataset provides 3D ground truth (GT) poses and video data. To run our implementation, you need to extract 2D poses from the video data using [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) and save the resulting pose data as `.npy` files.

### 3. Download Pretrained Models
We provide a pretrained diffusion model finetuned for the Yoga dataset.

* [yoga_diffusion_model.zip](https://drive.google.com/drive/folders/1SbvIOsJBMzPnREatkk6_MNNxGyoi3Ogw?usp=sharing)

After downloading, extract the archive into the `save/yoga_diffusion_model` folder.

---

## Usage for 3DHPE
To perform 3D Human Pose Estimation, place one of the 2D pose files (`.npy`) obtained from the 3DYoga90 dataset into the `dataset/nba/motions` directory, then run the following command:

```shell
python -m sample.mas \
  --model_path save/yoga_diffusion_model/checkpoint_200000.pth \
  --num_samples 1 \
  --seed 75 \
  --overwrite \
  --output_dir results \
  --use_data \
  --show_input_motions \
  --num_views 7 \
  --input_iterations 100
```

For more details on available flags and advanced configurations, please refer to the documentation of the original [MAS](https://github.com/roykapon/MAS) repository.

---

## Acknowledgments
 We built our codebase upon the [MAS (Multi-view Ancestral Sampling)](https://github.com/roykapon/MAS) repository developed by **Roy Kapon et al.** We sincerely thank the original authors.

---

## Citation
If you find our work useful for your research, please consider citing our FG 2026 paper:

<!-- ```bibtex
@inproceedings{C-MAS,
  title={Unsupervised 3D Human Pose Estimation via Conditional Multi-view Ancestral Sampling},
  author={Goto, Ryohei and },
  booktitle={Proceedings of the IEEE International Conference on Automatic Face and Gesture Recognition (FG)},
  year={2026}
} -->