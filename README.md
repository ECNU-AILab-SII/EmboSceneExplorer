
![Teaser](assets/teaser.png)

# EmboSceneExplorer: Embodied Scene Explorer for Multimodal Perception and Navigation

## 🔥 News
- [2025-7-22] Version 1.0 released! 🎉

## Table of Contents

1. [What is EmboSceneExplorer?](#what-is-EmboSceneExplorer)
2. [Key Features](#key-features)
3. [Quick Start](#quick-start)
4. [Support](#support)
5. [License and Acknowledgments](#license-and-acknowledgments)
6. [Citation](#citation)

## What is EmboSceneExplorer?

**EmboSceneExplorer** is a multimodal scene perception, understanding, and navigation system built on the Habitat simulation environment. It enables Embodied AI Agents to perform 3D perception and reconstruction, LLM-based grounding, and goal-oriented navigation within virtual 3D scenes (e.g., ScanNet, Matterport3D). The workflow comprises four core components:

1. **Multimodal Data Collection and Reconstruction**  
  Captures RGB/RGB-D image sequences, depth maps, and semantic maps with COLMAP-style camera poses, generating multimodal scene representations including:
   - High-fidelity meshes  
   - Dense point clouds  
   - Occupancy grid maps (Occ)

2. **3D Visual Grounding**  
   Bridges language and spatial understanding by:
   - Parsing natural language instructions (supporting both English and Chinese) into actionable goals
   - Grounding semantic concepts to 3D locations
   - Generating point-cloud-level accurate object masks from textual queries

4. **Autonomous Navigation**  
   Integrates scene representations (3DGS/Mesh/Occ) to:
   - Build navigable topological maps  
   - Plan optimal collision-free paths  
   - Execute exploration and goal-reaching behaviors

## Quick Start

### Prerequisites
- **Miniconda/Anaconda** (latest version)
- **NVIDIA GPU** (recommended for full performance)
- **Linux** (Ubuntu 20.04/22.04 recommended)

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
# Download example scenes:
gdown https://drive.google.com/file/d/1jwboFEruYFIG9c31qWga6X-vgbraKIt-/view?usp=sharing
unzip scenes.zip -d example_data/
cp example_data/scanet/pointnav_scannet.yaml ./submodules/habitat-lab/habitat-lab/habitat/config/benchmark/nav/pointnav

# Modify the root_path in example_data/scanet/scannet.yaml to the project's absolute path.
cp example_data/scanet/scannet.yaml ./submodules/habitat-lab/habitat-lab/habitat/config/habitat/dataset/pointnav/
```

### Start
```bash
cd bash scripts

# 1. Data collection:
bash data_collection.sh

# 2. RGB-D Reconstruction:
bash reconstruction.sh

# 3. Occupancy map reconstruction:
bash occupancy.sh

# 4. Visual grounding:
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
  author = {EmboSceneExplorer Authors},
  title = {EmboSceneExplorer: Embodied Scene Explorer for Multimodal Perception and Navigation},
  month = {July},
  year = {2025},
  url = {https://github.com/ECNU-AILab-SII/EmboSceneExplorer/}
}
