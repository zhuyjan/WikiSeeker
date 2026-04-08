<p align="center">
  <img width="400" height="300" src="figures/WikiSeeker_Logo.png">
</p>

# WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering

[![arXiv](https://img.shields.io/badge/arXiv-2604.05818-B21A1B)](https://arxiv.org/abs/2604.05818)
[![CVPR 2026](https://img.shields.io/badge/ACL%202026-Findings-blue)](https://2026.aclweb.org/)
[![GitHub Stars](https://img.shields.io/github/stars/zhuyjan/WikiSeeker?style=social)](https://github.com/zhuyjan/WikiSeeker/stargazers)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 🔍 Overview
<p align="center"><img src="figures/pipeline.png" alt="method" width="1000px" /></p>

We introduce WikiSeeker, a novel multi-modal RAG framework that bridges these gaps by proposing a multi-modal retriever and redefining the role of VLMs. Rather than serving merely as answer generators, we assign VLMs two specialized agents: a Refiner and an Inspector. The Refiner utilizes the capability of VLMs to rewrite the textual query according to the input image, significantly improving the performance of the multimodal retriever. The Inspector facilitates a decoupled generation strategy by selectively routing reliable retrieved context to another LLM for answer generation, while relying on the VLM’s internal knowledge when retrieval is unreliable.

## 🎯 Todo List
- [x] Release paper on Arxiv.
- [ ] Publish the details of dataset processing.
- [ ] Release the multi-modal retrieval code along with the corresponding knowledge base.
- [ ] Release the RL training code for Refiner.

## 🧭 Acknowledgements
Our code is built upon [EchoSight](https://github.com/Go2Heart/EchoSight), [OMGM](https://github.com/ChaoLinAViy/OMGM) and [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval). Thanks for their great work.
