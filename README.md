# Hierarchical Dense Correlation Distillation for Few-Shot Segmentation

This is the official implementation for our **CVPR 2023** [paper](https://arxiv.org/abs/2303.14652) "*Hierarchical Dense Correlation Distillation for Few-Shot Segmentation*".

> **📢Temporary Statement !!!**:
Thank you for your interest in our work! We have received many requests regarding our code and are excited to announce that we have released the raw code and [models](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155186045_link_cuhk_edu_hk/Eo5I56lRAOlIrKqcpFvA7NYBvvhR3QI8Gn_KLDdn9bb95A?e=YudoGF). However, please note that we have not yet provided any supplementary explanations. We will be releasing the reproduction guidance soon, so please stay tuned! 💻👀

🔬 You can start by reproducing our work based on our code. Please note that the experimental results may vary due to different environments and settings, which may sometimes lead to higher mIoU results than reported in the paper by up to 1.0%. However, it is still acceptable to compare your results with those reported in the paper.

> **Abstract:** *Few-shot semantic segmentation (FSS) aims to form class-agnostic models segmenting unseen classes with only a handful of annotations. Previous methods limited to the semantic feature and prototype representation suffer from coarse segmentation granularity and train-set overfitting. In this work, we design Hierarchically Decoupled Matching Network (HDMNet) mining pixel-level support correlation based on the transformer architecture. The self-attention modules are used to assist in establishing hierarchical dense features, as a means to accomplish the cascade matching between query and support features. Moreover, we propose a matching module to reduce train-set overfitting and introduce correlation distillation leveraging semantic correspondence from coarse resolution to boost fine-grained segmentation. Our method performs decently in experiments. We achieve 50.0% mIoU on \coco~dataset one-shot setting and 56.0% on five-shot segmentation, respectively.*


## BibTeX

If you find our work and this repository useful. Please consider giving a star :star: and citation &#x1F4DA;.

```bibtex
@article{peng2023hierarchical,
  title={Hierarchical Dense Correlation Distillation for Few-Shot Segmentation},
  author={Peng, Bohao and Tian, Zhuotao and Wu, Xiaoyang and Wang, Chenyao and Liu, Shu and Su, Jingyong and Jia, Jiaya},
  journal={arXiv preprint arXiv:2303.14652},
  year={2023}
}
```