Semi-Supervised Learning for Multi-Label Cardiovascular Diseases Prediction: A Multi-Dataset Study
=
* A Pytorch implementation of our under reviewed paper 
Semi-Supervised Learning for Multi-Label Cardiovascular Diseases Prediction: A Multi-Dataset Study
* [arxiv](https://arxiv.org/abs/2306.10494)
# Installation:

* Python 3.7
* Pytorch 1.3.1
* NVIDIA CUDA 9.2
* Numpy 1.20.3
* Scikit-learn 0.23.2
* scipy 1.3.1

# Preliminaries
* Prepare dataset: [SEED](https://bcmi.sjtu.edu.cn/~seed/index.html) and [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/index.html)

# Training 
* EEGMatch model definition file: model_EEGMatch.py 
* Pipeline of the EEGMatch: implementation_EEGMatch.py
* implementation of domain adversarial training: Adversarial_DG.py

# Citation
@misc{zhou2023semisupervised,
      title={Semi-Supervised Learning for Multi-Label Cardiovascular Diseases Prediction:A Multi-Dataset Study}, 
      author={Rushuang Zhou and Lei Lu and Zijun Liu and Ting Xiang and Zhen Liang and David A. Clifton and Yining Dong and Yuan-Ting Zhang},
      year={2023},
      eprint={2306.10494},
      archivePrefix={arXiv},
      primaryClass={eess.SP}
}
