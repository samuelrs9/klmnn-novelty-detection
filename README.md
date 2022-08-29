# klmnn-novelty-detection

This repository contains the Matlab implementation of the paper ["On novelty detection for multi-class classification using non-linear metric learning"](https://doi.org/10.1016/j.eswa.2020.114193).

## Experiments
To replicate the experiments described in the paper we have made scripts available in the ```experiments```` directory. The scripts
```
expressions/main_libras.m
expressions/main_iris.m
expressions/main_pose.m
expressions/main_glass.m
````
are used to run novelty detection experiments on real datasets.
On the other hand, the scripts
````
expressions/main_sim_1.m
expressions/main_sim_2.m
expressions/main_sim_3.m
expressions/main_sim_4.m
```
are used to run the experiments in the study simulations on synthetic datasets.

## Dependencies
* Large Margin Nearest Neighbors (LMNN)
* Kernel Principal Component Analysis (KPCA)

## Usage


## Compared Methods
* Kernel Null Foley-Sammon Transform (KNFST)
* One Class SVM
* Multi Class SVM
* Kernel Principal Component Analysis (KPCA)

## Citation
If you find our work useful for your research, please cite our paper:
```
@article{silva2021novelty,
  title={On novelty detection for multi-class classification using non-linear metric learning},
  author={Silva, Samuel Rocha and Vieira, Thales and Mart{\'\i}nez, Dimas and Paiva, Afonso},
  journal={Expert Systems with Applications},
  volume={167},
  pages={114193},
  year={2021},
  publisher={Elsevier}
}
```