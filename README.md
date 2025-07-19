# Squeeze and Excitation: A Weighted Graph Contrastive Learning for Collaborative Filtering


## Introduction

This is the Pytorch implementation for our WeightedGCL paper:

>Squeeze and Excitation: A Weighted Graph Contrastive Learning for Collaborative Filtering

## Environment Requirement

- python 3.9
- Pytorch 2.1.0



## Dataset

We provide three datasets: **Amazon Books, Pinterest, Alibaba**. These three datasets' contents are in the `dataset/` folder.

## Training

  ```
python main.py
  ```

## Performance Comparison

<img src="pic\performance.png"/>

## Citing WeightedGCL

If you find WeightedGCL useful in your research, please consider citing our [paper]().

```
@inproceedings{chen2025squeeze,
  title={Squeeze and Excitation: A Weighted Graph Contrastive Learning for Collaborative Filtering},
  author={Chen, Zheyu and Xu, Jinfeng and Wei, Yutong and Peng, Ziyue},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2769--2773},
  year={2025}
}
```

The code is released for academic research use only. For commercial use, please contact [Zheyu Chen](zheyu.chen@connect.polyu.hk).

## Acknowledgement

The structure of this code is based on [Recbole-GNN](https://github.com/RUCAIBox/RecBole-GNN). Thanks for their work.

