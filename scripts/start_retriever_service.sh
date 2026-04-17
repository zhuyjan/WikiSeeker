#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python utils/retriever_service.py \
    --host 0.0.0.0 \
    --port 5678 \
    --knowledge_base /data/zhuyingjian/rag/dataset/EVQA/encyclopedic_kb_wiki.json \
    --faiss_index /data/zhuyingjian/rag/dataset/EVQA/eva_qwen3_faiss_index/ \
    --retriever_vit eva-clip \
    --text_model_path /data/share/model/Qwen/Qwen3-Embedding-0.6B \
    --faiss_gpu_ids 0 \
    --vis_model_device 0 \
    --text_model_device 0 \
