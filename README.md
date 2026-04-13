# SISTA-Net: Compressive Sensing-Inspired Self-Supervised Single-Pixel Imaging
[![arXiv](https://img.shields.io/badge/arXiv-2603.29732-b31b1b.svg)](https://arxiv.org/abs/2603.29732)
[![GitHub](https://img.shields.io/badge/GitHub-JijunLu04%2FSISTA--Net-181717?logo=github)](https://github.com/JijunLu04/SISTA-Net)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C?logo=pytorch)](https://pytorch.org/)

Official PyTorch implementation of the paper **"Compressive Sensing Inspired Self-Supervised Single-Pixel Imaging"** .

---

## 📑 Pipeline
![SISTA-Net Pipeline](Figs/pipeline.png)
The overall pipeline of our proposed SISTA\-Net, integrating compressive sensing theory with self-supervised networks for physically interpretable reconstruction.

---

## 🏗️ Model Architecture

![Res2MM-Net Architecture](Figs/Res2MMNet.png)
A multi-scale residual network with VSSM blocks for efficient local-global feature modeling.

---

## 🧪 Simulation Experiments
![Simulation Results](Figs/ex_simu.gif)

Comparison of reconstruction performance across different sampling rates.

---

## 🚢 Real-World Experiments
![Real-World Results](Figs/ex_real.png)
Experimental results on real-world underwater environment.

---

## 📖 Citation
If you find our work useful, please consider citing:

```bibtex
@article{lu2026compressive,
  title={Compressive sensing inspired self-supervised single-pixel imaging},
  author={Lu, Jijun and Chen, Yifan and Chen, Libang and Zhou, Yiqiang and Zheng, Ye and Chen, Mingliang and Sun, Zhe and Li, Xuelong},
  journal={arXiv preprint arXiv:2603.29732},
  year={2026}
}