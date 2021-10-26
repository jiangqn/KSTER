#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

transformer_base_path=checkpoints/transformer
database_base_path=new_database

mkdir $database_base_path
for domain in law medical koran it subtitles
do
    mkdir $database_base_path/${domain}_en_de_base
    python3 -m joeynmt build_database configs/transformer_base_${domain}_en2de.yaml \
        --ckpt $transformer_base_path/wmt14_en_de.transformer.ckpt \
        --division train \
        --index_path $database_base_path/${domain}_en_de_base/trained.index \
        --token_map_path $database_base_path/${domain}_en_de_base/token_map \
        --embedding_path $database_base_path/${domain}_en_de_base/embeddings.npy
done

for domain in law medical koran it subtitles
do
    mkdir $database_base_path/${domain}_de_en_base
    python3 -m joeynmt build_database configs/transformer_base_${domain}_de2en.yaml \
        --ckpt $transformer_base_path/wmt14_de_en.transformer.ckpt \
        --division train \
        --index_path $database_base_path/${domain}_de_en_base/trained.index \
        --token_map_path $database_base_path/${domain}_de_en_base/token_map \
        --embedding_path $database_base_path/${domain}_de_en_base/embeddings.npy
done