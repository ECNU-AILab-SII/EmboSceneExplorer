<div align="center">

![Teaser](assets/teaser.png)

<h1> EmboSceneExplorer: Embodied Scene Explorer for Multimodal Perception and Navigation </h1>

Code Contributors: [Ao Gao](https://github.com/Yuhuoo), [Luosong Guo](https://max-luo-song.github.io/gls2000.github.io), [Chaoyang Li](https://github.com/GiIfoyle), [Jiangming Shi](https://shijiangming1.github.io), [Zilong Xie](https://github.com/XieZilongAI)  
Supervisors: [Jingyu Gong](https://jingyugong.github.io), [Xin Tan](https://tanxincs.github.io), [Zhizhong Zhang](https://faculty.ecnu.edu.cn/_s16/zzz2/main.psp), [Yuan Xie†](https://faculty.ecnu.edu.cn/_s16/xy2_11342/main.psp)(**Project Leader**)

[![README in English](https://img.shields.io/badge/English-d9d9d9)](./README.md)
[![简体中文版自述文件](https://img.shields.io/badge/简体中文-d9d9d9)](./README_CN.md)
</div>

## 🔥 News
- [2025-7-28] Version 1.0 released! 🎉

## Table of Contents

1. [What is EmboSceneExplorer?](#what-is-EmboSceneExplorer)
2. [Key Features](#key-features)
3. [Quick Start](#quick-start)
4. [Support](#support)
5. [License and Acknowledgments](#license-and-acknowledgments)
6. [Citation](#citation)

## What is EmboSceneExplorer?

**EmboSceneExplorer** is a multimodal scene perception, understanding, and navigation system built on the Habitat simulation environment. It enables Embodied AI Agents to perform 3D perception and reconstruction, LLM-based grounding, and goal-oriented navigation within virtual 3D scenes (e.g., ScanNet). The workflow comprises four core components:

1. **Multimodal Data Collection**  
   Captures multimodal data including:
   - RGB image sequences
   - Depth maps and semantic segmentation maps
   - COLMAP-style camera intrinsics and extrinsics (supporting 3D Gaussian Splatting training)
  
2. **Scene Reconstruction** 
   Builds multimodal scene representations including:
   - Dense point clouds  
   - High-fidelity meshes
   - Occupancy grid maps (Occ)

3. **3D Visual Grounding**  
   Bridging language and spatial understanding, we've developed a **3D visual grounding model** that currently achieves **state-of-the-art** performance across multiple metrics::
   - Parsing natural language instructions (supporting both English and Chinese) into actionable goals
   - Grounding semantic concepts to precise 3D locations
   - Generating point-cloud-level accurate object from textual queries

4. **Autonomous Navigation**  
   Integrates scene representations to:
   - Build navigable occupancy maps  
   - Plan optimal collision-free paths  
   - Execute exploration and goal-reaching behaviors

## Quick Start

### Prerequisites
- **Miniconda/Anaconda**
- **NVIDIA GPU** (CUDA 11.8)
- **Linux**

### Cloning the Repository
```bash
# HTTPS
git clone --recursive https://github.com/ECNU-AILab-SII/EmboSceneExplorer.git

or

# SSH
git clone --recursive git@github.com:ECNU-AILab-SII/EmboSceneExplorer.git
```

### Conda Environment Setup
```bash
# Create environment from provided YAML
conda env create -f environment.yml

# Activate environment
conda activate emboscene
```

### Data Preparation
```bash
# Download example scenes
gdown https://drive.google.com/file/d/1jwboFEruYFIG9c31qWga6X-vgbraKIt-/view?usp=sharing
unzip scenes.zip -d example_data/

# Modify the root_path in example_data/scanet/scannet.yaml to the project's absolute path
data_path: /xxx/xxx/EmboSceneExplorer/....

# Copy point.yaml and scant.yaml to corresponding locations in submodules  
cp example_data/scanet/pointnav_scannet.yaml ./submodules/habitat-lab/habitat-lab/habitat/config/benchmark/nav/pointnav
cp example_data/scanet/scannet.yaml ./submodules/habitat-lab/habitat-lab/habitat/config/habitat/dataset/pointnav/

# Download pretrained 3D visual grounding model
gdown https://drive.google.com/file/d/1OlBSTpcyIlcCqxqKgYztss6bBIIPJDFc/view?usp=sharing
```

### Start
```bash
cd bash_scripts

# 1. Data collection:
bash data_collection.sh

# 2. RGB-D Reconstruction:
bash reconstruction.sh

# 3. Occupancy map reconstruction:
bash occupancy.sh

# 4. 3D Visual grounding (supporting both English and Chinese):
bash visual_grounding.sh

# 5. Navigation:
bash navigation.sh
```

## Support

- Report bugs or request features via GitHub [Issues](https://github.com/ECNU-AILab-SII/EmboSceneExplorer/issues).
- Join discussions or ask questions on GitHub [Discussions](https://github.com/ECNU-AILab-SII/EmboSceneExplorer/discussions).

## License and Acknowledgments

EmboSceneExplorer is MIT licensed. See the [LICENSE](LICENSE) for details.

EmboSceneExplorer's development has been made possible thanks to these open-source projects:

- [Habitat-sim](https://github.com/facebookresearch/habitat-sim.git): A flexible, high-performance 3D simulator for embodied AI research.
- [Habitat-lab](https://github.com/facebookresearch/habitat-lab): A modular high-level library for end-to-end development in embodied AI.
- [3D-LLaVA](https://github.com/djiajunustc/3D-LLaVA): Towards Generalist 3D LMMs with Omni Superpoint Transformer.

## Citation

If you use EmboSceneExplorer in your research, please consider citing:

```bibtex
@misc{EmboSceneExplorer,
  author = {Ao Gao, Luosong Guo, Chaoyang Li, Jiangming Shi, Zilong Xie, Jingyu Gong, Xin Tan, Zhizhong Zhang, Yuan Xie†},
  title = {EmboSceneExplorer: Embodied Scene Explorer for Multimodal Perception and Navigation},
  month = {July},
  year = {2025},
  url = {https://github.com/ECNU-AILab-SII/EmboSceneExplorer/}
}
