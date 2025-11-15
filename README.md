# Efficient Neuron Segmentation in Electron Microscopy by Affinity-Guided Queries (ICLR 2025)

<p align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=chenhang98.AGQ" alt="访客统计" />
  <img src="https://img.shields.io/github/stars/chenhang98/AGQ?style=social" alt="GitHub stars" />
  <img alt="Static Badge" src="https://img.shields.io/badge/license-MIT-blue.svg" />
  <a href="https://openreview.net/pdf?id=Y0QqruhqIa"><img src="https://img.shields.io/badge/Paper-red.svg" alt="Paper"></a>
  <a href="assets/iclr25_poster.pdf"><img src="https://img.shields.io/badge/Poster-yellow.svg" alt="Poster"></a>
</p>

This is the official repo for ["Efficient Neuron Segmentation in Electron Microscopy by Affinity-Guided Queries"](https://openreview.net/pdf?id=Y0QqruhqIa).


## Introduction

Accurate segmentation of neurons in electron microscopy (EM) images plays a crucial role in understanding the intricate wiring patterns of the brain. Existing
automatic neuron segmentation methods rely on traditional clustering algorithms, where affinities are predicted first, and then watershed and post-processing algorithms are applied to yield segmentation results. Due to the nature of watershed algorithm, this paradigm has deficiency in both prediction quality and speed. Inspired by recent advances in natural image segmentation, we propose to use query-based methods to address the problem because they do not necessitate watershed algorithms. However, we find that directly applying existing query-based methods faces great challenges due to the large memory requirement of the 3D data and considerably different morphology of neurons. To tackle these challenges, we introduce affinity-guided queries and integrate them into a lightweight query-based framework. Specifically, we first predict affinities with a lightweight branch, which provides coarse neuron structure information. The affinities are then used to construct affinity-guided queries, facilitating segmentation with bottom-up cues. These queries, along with additional learnable queries, interact with the image features to directly predict the final segmentation results. Experiments on benchmark datasets demonstrated that our method achieved better results over state-of-the-art methods with a 2∼3× speedup in inference.

![image](assets/overview.png)


## Installation

```bash
pip install -r requirements.txt
```

This repo is based on [pytorch_connectomics](https://github.com/zudi-lin/pytorch_connectomics). Please refer to [pytorch_connectomics](https://github.com/zudi-lin/pytorch_connectomics) for more details.

## Inference and Evaluation

```bash
bash scripts/inference.sh
```
The model checkpoint is available at `checkpoints/checkpoint_200000.pth.tar`.

## Training

```bash
bash scripts/train.sh
```

## Acknowledgement

We would like to thank [pytorch_connectomics](https://github.com/zudi-lin/pytorch_connectomics) for providing the codebase.

## Citation

If you find this project useful in your research, please consider citing:

```
@inproceedings{AGQ,
  author       = {Hang Chen and
                  Chufeng Tang and
                  Xiao Li and
                  Xiaolin Hu},
  title        = {Efficient Neuron Segmentation in Electron Microscopy by Affinity-Guided
                  Queries},
  booktitle    = {The Thirteenth International Conference on Learning Representations,
                  {ICLR},
  year         = {2025}
}
```