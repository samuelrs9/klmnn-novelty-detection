# klmnn-novelty-detection

This repository contains the Matlab implementation of the paper ["On novelty detection for multi-class classification using non-linear metric learning"](https://doi.org/10.1016/j.eswa.2020.114193).

## Experiments
To replicate the experiments described in the paper we have made scripts available in the `experiments` directory. The scripts
```
experiments/main_libras.m
experiments/main_iris.m
experiments/main_pose.m
experiments/main_glass.m
```
are used to run novelty detection experiments on real datasets.
On the other hand, the scripts
```
experiments/main_sim_1.m
experiments/main_sim_2.m
experiments/main_sim_3.m
experiments/main_sim_4.m
```
are used to run the experiments in the simulation studies on synthetic datasets.

## External dependencies
* Large Margin Nearest Neighbors (LMNN)
* Kernel Null Foley-Sammon Transform (KNFST)
* Kernel Principal Component Analysis (KPCA)

## Compared Methods
We compare our approach with the following methods
* Kernel Null Foley-Sammon Transform (KNFST)
* One Class SVM
* Multi Class SVM
* Kernel Principal Component Analysis (KPCA)

Codes for these approaches are available in the `compared_methods` directory. For **One Class** and **Multi Class SVM** we use the implementation available o Matlab. In the case of the **KNFST** and **KPCA** methods, we use external implementations that are located in the `external` directory.

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