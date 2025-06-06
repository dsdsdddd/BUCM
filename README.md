<<<<<<< HEAD
# BUCM
=======
# Monocular Depth Estimation Model Based on Uncertainty Constraints and Its Application in Augmented Reality                                                                                                                              

## Overview

This repository contains the implementation of the monocular depth estimation model based on uncertainty constraints proposed in the paper “Monocular Depth Estimation Model Based on Uncertainty Constraints and Its Application in Augmented Reality”. The model fuses single-view and multi-view depth estimation techniques to improve depth prediction accuracy and robustness in augmented reality (AR), especially enhancing occlusion handling at object edges through Bayesian uncertainty modeling.

The model framework is shown in the figure：

<img src="./media/2.jpg" alt="模型框架图" width="1000"/>


### Key Features

- **Fusion of Single-view and Multi-view Depth Estimation**: Enhances multi-view depth cost volumes with features extracted from single-view encoders, substantially improving prediction accuracy in dynamic and complex scenes.
- **Bayesian Convolutional Uncertainty Estimation (CBM)**: Introduces uncertainty constraints to better model depth uncertainty, improving robustness in occlusion regions.
- **AR Assembly Training System**: Development of a mobile AR parts assembly training system that leverages the model to achieve precise virtual-real occlusion for interactive component assembly tasks.
- **Datasets**: Validated on the ScanNet dataset, showing superior performance over several mainstream depth estimation methods and enabling high-precision virtual-real occlusion in AR.
### Experimental result diagram.

- ** Depth estimation result diagram.**
<img src="./media/fig3.jpg" alt="深度图" width="600"/>

- ** Real-virtual occlusion effect diagram.**
<img src="./media/9.jpg" alt="遮挡图" width="600"/>
  
- ** 3D point cloud reconstruction diagram.**
<img src="./media/11.jpg" alt="3d图" width="600"/>
  
- ** AR application scenario diagram.**
<img src="./media/14.jpg" alt="AR图" width="600"/>

The evaluation metrics for the depth estimation performance of our model on the ScanNet dataset are shown in the table below.

| Methods | Abs Diff | ABS Rel | Sq Rel | \(\delta < 1.05\) | \(\delta < 1.25\) |
|----------|----------|----------|----------|----------|----------|
| Baseline | 0.0941 | 0.0457 | 0.0139 | 71.95 | 97.84 |
| Ours(without DEP) | 0.0864 | 0.0422 | 0.0115 | 73.83 | 98.14 |
| Ours | 0.0843 | 0.0395 | 0.0103 | 74.79 | 98.21 |

The evaluation metrics for the depth estimation performance of our model on the 7Scenes dataset are shown in the table below.

| Methods | Abs Diff | ABS Rel | Sq Rel | \(\delta < 1.05\) | \(\delta < 1.25\) |
|----------|----------|----------|----------|----------|----------|
| Baseline | 0.1045 | 0.0617 | 0.0175 | 57.30 | 97.02 |
| Ours(without DEP) | 0.1104 | 0.0599 | 0.0170 | 60.45 | 96.62 |
| Ours | 0.1095 | 0.0577 | 0.0161 | 61.03 | 96.94 |
  
The model demonstrates excellent performance in depth estimation, effectively handling occlusion issues, and shows great potential for applications in augmented reality (AR) scenarios.

## Installation

### Prerequisites

- Python 3.8+
- Conda (Anaconda or Miniconda)
- CUDA-enabled GPU (e.g., NVIDIA RTX3090)
- Optional: Segment Anything Model (SAM) for DEP module.

### Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/sdfsfwe/Virtual-Real-Occlusion.git
   cd Virtual-Real-Occlusion
   ```

2. **Create and activate the Conda environment**:

   The required dependencies are specified in the `env.yaml` file. To set up the environment:

   ```bash
   conda env create -f env.yaml
   conda activate multiview-depth
   ```

   This will install all necessary packages, including PyTorch, torchvision, NumPy, OpenCV, and others specified in `env.yaml`.

3. **Verify installation**:

   Ensure the environment is set up correctly by running:

   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

4. **Download and preprocess datasets** (ScanNetV2, 7Scenes):

   Follow instructions in `data/README.md` to set up the datasets.
   Please follow the instructions [here](https://github.com/ScanNet/ScanNet) to download the ScanNet dataset. This dataset is quite big (>2TB), so make sure you have enough space, especially for extracting files. Follow the instructions [here](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/?msockid=0faa059eb12a673618d31147b0f86613) to download the 7Scenes dataset.

## Training

   By default models and tensorboard event files are saved to `~/tmp/tensorboard/<model_name>`.
   This can be changed with the `--log_dir` flag.

   We train with a batch_size of 16 with 16-bit precision on two RTX3090 on the default ScanNetv2 split.

   Example command to train with two GPUs:

   ```shell
   CUDA_VISIBLE_DEVICES=0,1 python train.py --name HERO_MODEL \
            --log_dir logs \
            --config_file configs/models/hero_model.yaml \
            --data_config configs/data/scannet_default_train.yaml \
            --gpus 2 \
            --batch_size 16;
   ```

   The code supports any number of GPUs for training.
   You can specify which GPUs to use with the `CUDA_VISIBLE_DEVICES` environment.

   **Different dataset**

   You can train on a custom MVS dataset by writing a new dataloader class which inherits from `GenericMVSDataset` at `datasets/generic_mvs_dataset.py`. See the `ScannetDataset` class in `datasets/scannet_dataset.py` or indeed any other class in `datasets` for an example.

## Testing and Evaluation

You can use `test.py` for inferring and evaluating depth maps and fusing meshes. 

All results will be stored at a base results folder (results_path) at:

    opts.output_base_path/opts.name/opts.dataset/opts.frame_tuple_type/

where opts is the `options` class. For example, when `opts.output_base_path` is `./results`, `opts.name` is `HERO_MODEL`,
`opts.dataset` is `scannet`, and `opts.frame_tuple_type` is `default`, the output directory will be 

    ./results/HERO_MODEL/scannet/default/

Make sure to set `--opts.output_base_path` to a directory suitable for you to store results.

`--frame_tuple_type` is the type of image tuple used for MVS. A selection should 
be provided in the `data_config` file you used. 

By default `test.py` will attempt to compute depth scores for each frame and provide both frame averaged and scene averaged metrics. The script will save these scores (per scene and totals) under `results_path/scores`.

We've done our best to ensure that a torch batching bug through the matching 
encoder is fixed for (<10^-4) accurate testing by disabling image batching 
through that encoder. Run `--batch_size 4` at most if in doubt, and if 
you're looking to get as stable as possible numbers and avoid PyTorch 
gremlins, use `--batch_size 1` for comparison evaluation.

If you want to use this for speed, set `--fast_cost_volume` to True. This will
enable batching through the matching encoder and will enable an einops 
optimized feature volume.


```bash
# Example command to just compute scores 
CUDA_VISIBLE_DEVICES=0 python test.py --name HERO_MODEL \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/hero_model.yaml \
            --load_weights_from_checkpoint weights/hero_model.ckpt \
            --data_config configs/data/scannet_default_test.yaml \
            --num_workers 8 \
            --batch_size 4;

# If you'd like to get a super fast version use:
CUDA_VISIBLE_DEVICES=0 python test.py --name HERO_MODEL \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/hero_model.yaml \
            --load_weights_from_checkpoint weights/hero_model.ckpt \
            --data_config configs/data/scannet_default_test.yaml \
            --num_workers 8 \
            --fast_cost_volume \
            --batch_size 2;
```

This script can also be used to perform a few different auxiliary tasks.

## Point Cloud Fusion

We also allow point cloud fusion of depth maps using the fuser from 3DVNet's [repo](https://github.com/alexrich021/3dvnet/blob/main/mv3d/eval/pointcloudfusion_custom.py). 

```bash
# Example command to fuse depths into point clouds.
CUDA_VISIBLE_DEVICES=0 python pc_fusion.py --name HERO_MODEL \
            --output_base_path OUTPUT_PATH \
            --config_file configs/models/hero_model.yaml \
            --load_weights_from_checkpoint weights/hero_model.ckpt \
            --data_config configs/data/scannet_dense_test.yaml \
            --num_workers 8 \
            --batch_size 4;
```

Change `configs/data/scannet_dense_test.yaml` to `configs/data/scannet_default_test.yaml` to use keyframes only if you don't want to wait too long.

## Acknowledgements

We would like to express our sincere gratitude to Liu Jia, Wang Bin, Chen Dapeng, Song Hong, Zhang Zengwei, and Huang Nanxuan from Nanjing University of Information Science and Technology for their invaluable support in both the code development and the paper. Their contributions have been crucial in advancing this project, and we greatly appreciate their guidance and collaboration.

>>>>>>> cce9ab7 (first commit)
