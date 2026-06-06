# MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration

[![Paper](https://img.shields.io/badge/arXiv-2506.19835-b31b1b.svg)](https://arxiv.org/abs/2506.19835)
[![GitHub](https://img.shields.io/badge/GitHub-MAM-blue?logo=github)](https://github.com/yczhou001/MAM)

<p align="center"> <img src="images/pipeline.jpg" width="100%"> </p>




## 📰 News
- `2025.06.05` [Project](https://github.com/yczhou001/MAM) is released.
- `2025.05.16` [MAM](https://www.arxiv.org/abs/2506.19835) is accepted as ACL 2025 (findings).

## 🔥 Abstract

Recent advancements in medical Large Language Models (LLMs) have showcased their powerful reasoning and diagnostic capabilities. Despite their success, current unified multimodal medical LLMs face limitations in knowledge update costs, comprehensiveness, and flexibility. To address these challenges, we introduce the Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis (MAM). Inspired by our empirical findings highlighting the benefits of role assignment and diagnostic discernment in LLMs, MAM decomposes the medical diagnostic process into specialized roles: a General Practitioner, Specialist Team, Radiologist, Medical Assistant, and Director, each embodied by an LLM-based agent. This modular and collaborative framework enables efficient knowledge updates and leverages existing medical LLMs and knowledge bases. Extensive experimental evaluations conducted on a wide range of publicly accessible multimodal medical datasets, incorporating text, image, audio, and video modalities, demonstrate that MAM consistently surpasses the performance of modality-specific LLMs. Notably, MAM achieves significant performance improvements ranging from 18% to 365% compared to baseline models. For more details, please refer to the [paper](https://www.arxiv.org/abs/2506.19835).

## ✨ Highlights

* We evaluate our MAM framework in **multimodal medical diagnosis tasks** through comprehensive experiments on **several publicly available multimodal medical datasets**.
* Experimental results demonstrate that the MAM framework consistently **outperforms specific-modal LLMs** across various medical datasets and data modalities. 
* We conduct **ablation studies**, **consistency analysis**, and **sensitivity analyses** regarding the number of discussion rounds and roles to gain deeper insights into the roles of individual components and the operational mechanisms of the framework.

## 😮 Experimental results
* Ablation study of our MAM framework. The “Direct” represents the baseline. From left to right, we incrementally add functions. “+Retrieval” is our full MAM framework.
<p align="center"> <img src="images/table7.jpg" width="50%"> </p>

For more details, please refer to the [paper](https://openreview.net/forum?id=NpTCAExiu3).

## 🙌 Contents
- [Install](#install)
- [Use Chat API](#use-chat-api-text-image-video-audio)
- [Quick Start](#quick-start-end-to-end-pipeline)


## 🛠️ Install
1. Clone this repository and navigate to MAM folder
```bash
git clone https://github.com/yczhou001/MAM.git
cd MAM
```
2. Install Package
```Shell
conda create -n mam python=3.10 -y
conda activate mam
pip install --upgrade pip  # enable PEP 660 support
pip install fire torch torchvision transformers==4.37.2 decord imageio==2.34.0 imageio-ffmpeg==0.4.9 moviepy==1.0.3 einops attrdict timm==0.9.16 scenedetect==0.6.3 opencv-python transformers_stream_generator tiktoken sentencepiece accelerate
```

## Use Chat API (Text, Image, Video, Audio)
```
python -m use_api
```

## Quick Start (End-to-End Pipeline)
```bash
python -m inference --step_id 9 --question 'What does the picture show?' --file_name tmp/image.png
```

<p align="center"> <img src="images/case_study1.jpg" width="90%"> </p>




## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@inproceedings{zhou-etal-2025-mam,
    title = "{MAM}: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration",
    author = "Zhou, Yucheng  and
      Song, Lingran  and
      Shen, Jianbing",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1298/",
    doi = "10.18653/v1/2025.findings-acl.1298",
    pages = "25319--25333",
    ISBN = "979-8-89176-256-5"
}
```
