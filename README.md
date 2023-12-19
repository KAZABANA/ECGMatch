Semi-Supervised Learning for Multi-Label Cardiovascular Diseases Prediction: A Multi-Dataset Study
=
* A Pytorch implementation of our under-reviewed paper 
Semi-Supervised Learning for Multi-Label Cardiovascular Diseases Prediction: A Multi-Dataset Study
* [IEEE TPAMI](https://ieeexplore.ieee.org/document/10360273)
# Preliminaries
* Physionet datasets: [Physionet datasets]([https://bcmi.sjtu.edu.cn/~seed/index.html](https://physionet.org/content/challenge-2021/1.0.3/))
* quick download: wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/
* Supplementary materials for the paper published in IEEE: SupplementaryMaterial_ECGMatch.pdf
# Training 
* model definition file: model_code.py 
* main function: main.py
* data preprocess and loading: dataloading.py
* algorithm: training_code.py
# Citation
@ARTICLE{10360273,
  author={Zhou, Rushuang and Lu, Lei and Liu, Zijun and Xiang, Ting and Liang, Zhen and Clifton, David A. and Dong, Yining and Zhang, Yuan-Ting},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Semi-Supervised Learning for Multi-Label Cardiovascular Diseases Prediction: A Multi-Dataset Study}, 
  year={2023},
  volume={},
  number={},
  pages={1-17},
  doi={10.1109/TPAMI.2023.3342828}}
