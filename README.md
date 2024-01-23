<div align="center">
<h1>
<b>
Text-Video Retrieval with Global-Local Semantic Consistent Learning
</b>
</h1>
<h4>
<a href="https://github.com/zchoi">Haonan Zhang</a>, <a href="https://ppengzeng.github.io/">Pengpeng Zeng</a>, <a href="https://lianligao.github.io/">Lianli Gao</a>, <a href="https://cfm.uestc.edu.cn/~songjingkuan/">Jingkuan Song</a>, Yihang Duan, <a href="https://scholar.google.com/citations?hl=zh-CN&user=kVcO9R4AAAAJ&view_op=list_works&sortby=pubdate">Xinyu Lyu</a>, <a href="https://cfm.uestc.edu.cn/~shenht/">Heng Tao Shen</a>, 
</h4>

[Paper] | **IJCAI 2024** 
</div>
This is the code implementation of the paper "Text-Video Retrieval with Global-Local Semantic Consistent Learning", the checkpoint and feature will be released soon.

## Overview
Adapting large-scale image-text pre-training models, e.g., CLIP, to the video domain represents the current state-of-the-art for text-video retrieval. The primary approaches involve transferring text-video pairs to a common embedding space and leveraging cross-modal interactions on specific entities for semantic alignment. Though effective, these paradigms entail prohibitive computational costs, leading to inefficient retrieval. To address this, we propose a simple yet effective method, Global-Local Semantic Consistent Learning (GLSCL),which capitalizes on latent shared semantics across modalities for text-video retrieval. Specifically, we introduce a parameter-free global interaction module to explore coarse-grained alignment. Then, we devise a shared local interaction module that employs several learnable queries to capture latent semantic concepts for learning fine-grained alignment. Moreover, we propose an inter-consistency loss and an intra-diversity loss to ensure the similarity and diversity of these concepts across and within modalities,respectively.

<p align="center">
    <img src=imgs/introduction.png width="70%"><br>
    <span><b>Figure 1. Performance comparison of the retrieval results (R@1) and computational costs (FLOPs) for text-to-video retrieval models.</b></span>
</p>



## Method

<p align="center">
    <img src=imgs/framework.png><br>
    <span><b>Figure 2. Overview of the proposed GLSCL for Text-Video retrieval.</b></span>
</p>
