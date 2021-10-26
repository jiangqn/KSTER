#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

transformer_base_path=checkpoints/transformer
combiner_base_path=checkpoints/combiner
database_base_path=database
analysis_base_path=analysis

mkdir $analysis_base_path
for domain in law medical koran it subtitles
do
    python3 -m joeynmt analyze configs/transformer_base_${domain}_en2de.yaml \
        --ckpt $transformer_base_path/wmt14_en_de.transformer.ckpt \
        --combiner dynamic_combiner \
        --combiner_path $combiner_base_path/${domain}_en_de.laplacian.combiner.ckpt \
        --top_k 16 \
        --kernel laplacian \
        --index_path $database_base_path/${domain}_en_de_base/trained.index \
        --token_map_path $database_base_path/${domain}_en_de_base/token_map \
        --embedding_path $database_base_path/${domain}_en_de_base/embeddings.npy \
        --in_memory False \
        --output_path $analysis_base_path/${domain}_en_de 
done