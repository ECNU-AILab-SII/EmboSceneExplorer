<div align="center">

![效果图](assets/teaser.png)

<h1> EmboSceneExplorer: 具身多模态感知与导航探索平台 </h1>

Code Contributors: [高傲](https://github.com/Yuhuoo), [郭洛松](https://max-luo-song.github.io/gls2000.github.io/), [李超阳](https://github.com/GiIfoyle), [施江鸣](https://shijiangming1.github.io), [谢子龙](https://github.com/XieZilongAI)  
Supervisors: [龚靖渝](https://jingyugong.github.io), [谭鑫](https://tanxincs.github.io), [张志忠](https://faculty.ecnu.edu.cn/_s16/zzz2/main.psp), [谢源†](https://faculty.ecnu.edu.cn/_s16/xy2_11342/main.psp)(**Project Leader**)

[![README in English](https://img.shields.io/badge/English-d9d9d9)](./README.md)
[![简体中文版自述文件](https://img.shields.io/badge/简体中文-d9d9d9)](./README_CN.md)
</div>

## 🔥 最新动态
- [2025年7月28日] 版本 1.0 正式发布！🎉

## 目录

1. [什么是EmboSceneExplorer?](#什么是EmboSceneExplorer)
2. [核心功能](#核心功能)
3. [快速开始](#快速开始)
4. [支持](#支持)
5. [许可与致谢](#许可与致谢)
6. [引用](#引用)

## 什么是EmboSceneExplorer?

**EmboSceneExplorer** 是一个基于 Habitat 仿真环境构建的多模态场景感知、理解与导航系统。它使具身人工智能代理能够在虚拟3D场景（如 ScanNet）中执行3D感知与重建、基于大语言模型（LLM）的定位以及目标导向的导航。其工作流程包含四个核心组件：

1. **多模态数据采集**  
   采集多模态数据，包括：
   - RGB图像序列
   - 深度图和语义图
   - Colmap格式的相机内外参（支持3DGS训练）

2. **场景重建**
   多模态场景表示重建，包括：
   - 稠密点云
   - 高保真网格模型
   - 占据栅格地图（Occ）

3. **3D视觉定位**  
   连接语言与空间理解，我们开发了一个**3D visual grounding模型**，目前各项指标达到**SOTA**：
   - 将自然语言指令（支持中英文）解析为可执行目标
   - 将语义概念定位到3D空间位置
   - 根据文本查询生成点云级别的精确物体

4. **自主导航**  
   集成场景表示（3DGS/网格/Occ）实现：
   - 构建可导航的占据栅格地图
   - 规划最优无碰撞路径
   - 执行探索与目标抵达行为

## 快速开始

### 前提条件
- **Miniconda/Anaconda**
- **NVIDIA GPU**（CUDA 11.8）
- **Linux**

### 克隆仓库
```bash
# HTTPS
git clone --recursive https://github.com/ECNU-AILab-SII/EmboSceneExplorer.git

或

# SSH
git clone --recursive git@github.com:ECNU-AILab-SII/EmboSceneExplorer.git
```

### Conda环境设置
```bash
# 使用提供的YAML文件创建环境
conda env create -f environment.yml

# 激活环境
conda activate emboscene
```

### 数据准备
```bash
# 下载测试场景:
gdown https://drive.google.com/file/d/1jwboFEruYFIG9c31qWga6X-vgbraKIt-/view?usp=sharing
unzip scenes.zip -d example_data/

# 修改example_data/scanet/scannet.yaml中root_path指向当前项目的绝对路径
data_path: /xxx/xxx/EmboSceneExplorer/....

# 拷贝此yaml到子模块的对应位置中
cp example_data/scanet/pointnav_scannet.yaml ./submodules/habitat-lab/habitat-lab/habitat/config/benchmark/nav/pointnav
cp example_data/scanet/scannet.yaml ./submodules/habitat-lab/habitat-lab/habitat/config/habitat/dataset/pointnav/

# 下载预训练的3D Visual Grounding权重
gdown https://drive.google.com/file/d/1OlBSTpcyIlcCqxqKgYztss6bBIIPJDFc/view?usp=sharing
```

### 开始
```bash
cd bash_scripts

# 1. 数据采集:
bash data_collection.sh

# 2. RGB-D 重建:
bash reconstruction.sh

# 3. Occupancy建图:
bash occupancy.sh

# 4. 执行3D Visual Grounding（支持中英文）:
bash visual_grounding.sh

# 5. 导航到目标位置:
bash navigation.sh
```

## 支持
- 通过 [Issues](https://github.com/ECNU-AILab-SII/EmboSceneExplorer/issues) 报告Bug或请求新功能。

- 在 [Discussions](https://github.com/ECNU-AILab-SII/EmboSceneExplorer/discussions) 参与讨论或提问。


## 许可与致谢

EmboSceneExplorer 采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

EmboSceneExplorer 的开发得益于以下开源项目：

- [Habitat-sim](https://github.com/facebookresearch/habitat-sim.git)：一个灵活、高性能的具身AI研究3D仿真器。

- [Habitat-lab](https://github.com/facebookresearch/habitat-lab)：用于具身AI端到端开发的模块化高级库。

- [3D-LLaVA](https://github.com/djiajunustc/3D-LLaVA)：基于全向超点变换器（Omni Superpoint Transformer）的通用3D大语言模型（LMM）。

## 引用

如果您在研究中使用了EmboSceneExplorer，请考虑引用：
```bibtex
@misc{EmboSceneExplorer,
  author = {Ao Gao, Luosong Guo, Chaoyang Li, Jiangming Shi, Zilong Xie, Jingyu Gong, Xin Tan, Zhizhong Zhang, Yuan Xie},
  title = {EmboSceneExplorer: Embodied Scene Explorer for Multimodal Perception and Navigation},
  month = {July},
  year = {2025},
  url = {https://github.com/ECNU-AILab-SII/EmboSceneExplorer/}
}