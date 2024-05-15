<div align="center">
<h1>
<b>
Text-Video Retrieval with Global-Local Semantic Consistent Learning
</b>
</h1>
<h4>
<a href="https://zchoi.github.io/">Haonan Zhang</a>, <a href="https://ppengzeng.github.io/">Pengpeng Zeng</a>, <a href="https://lianligao.github.io/">Lianli Gao</a>, <a href="https://cfm.uestc.edu.cn/~songjingkuan/">Jingkuan Song</a>, Yihang Duan, <a href="https://xinyulyu.github.io/">Xinyu Lyu</a>, <a href="https://cfm.uestc.edu.cn/~shenht/">Heng Tao Shen</a>, 

</h4>

</div>
This is the code implementation of the paper "Text-Video Retrieval with Global-Local Semantic Consistent Learning", the checkpoint and feature will be released soon.

## 🔥Updates

- [ ] Release the pre-trained weight and datasets.
- [x] Release the training and evaluation code.

## ✨Overview
Adapting large-scale image-text pre-training models, e.g., CLIP, to the video domain represents the current state-of-the-art for text-video retrieval. The primary approaches involve transferring text-video pairs to a common embedding space and leveraging cross-modal interactions on specific entities for semantic alignment. Though effective, these paradigms entail prohibitive computational costs, leading to inefficient retrieval. To address this, we propose a simple yet effective method, Global-Local Semantic Consistent Learning (GLSCL), which capitalizes on latent shared semantics across modalities for text-video retrieval. Specifically, we introduce a parameter-free global interaction module to explore coarse-grained alignment. Then, we devise a shared local interaction module that employs several learnable queries to capture latent semantic concepts for learning fine-grained alignment. Moreover, we propose an inter-consistency loss and an intra-diversity loss to ensure the similarity and diversity of these concepts across and within modalities, respectively.

<p align="center">
    <img src=imgs/introduction.png width="70%"><br>
    <span><b>Figure 1. Performance comparison of the retrieval results (R@1) and computational costs (FLOPs) for text-to-video retrieval models.</b></span>
</p>



## 🍀Method
Overview of the proposed GLSCL for text-video retrieval. It comprises two main components: (1) Global Interaction Module (GIM) captures coarse-level semantic information among text and video data without involving trainable parameters, and (2) Local Interaction Module (LIM) achieves fine-grained alignment within a shared latent semantic space via several lightweight queries. Furthermore, we introduce an inter-consistency loss and an intra-diversity loss to guarantee consistency and diversity of the shared semantics across and within modalities, respectively.
<p align="center">
    <img src=imgs/framework.png><br>
    <span><b>Figure 2. Overview of the proposed GLSCL for Text-Video retrieval.</b></span>
</p>

## 🧪Experiments
TODO


## 📚 Citation

```bibtex
@inproceedings{GLSCL,
  author    = {Haonan Zhang and
              Pengpeng Zeng and
              Lianli Gao and
              Jingkuan Song and
              Yihang Duan and
              Xinyu Lyu and
              Hengtao Sheng
            },
  title     = {Text-Video Retrieval with Global-Local Semantic Consistent Learning},
  booktitle = {Arxiv},
  year      = {2024}
}
```
