#!/bin/bash

python run_retrieval_with_service.py \
    --sample_file /data/zhuyingjian/rag/dataset/EVQA/test.csv \
    --query_expansion query/evqa_test_query_final.json \
    --retriever_service_url http://localhost:5678 \
    --top_ks 1,5,10,20 \
    --retrieval_top_k 20 \
    --alpha 0.59 \
    --save_result_path ret_res/evqa_ret_res.json
