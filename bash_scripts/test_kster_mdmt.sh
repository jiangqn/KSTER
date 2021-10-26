#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

transformer_base_path=checkpoints/transformer
combiner_base_path=checkpoints/combiner
database_base_path=multi_domain_database

echo "test KSTER multi domain performance in DAMT setting in EN-DE direction"
for domain in wmt14 law medical koran it subtitles
do
echo $domain
python3 -m joeynmt test configs/transformer_base_${domain}_en2de.yaml \
     --ckpt $transformer_base_path/wmt14_en_de.transformer.ckpt \
     --combiner dynamic_combiner \
     --combiner_path $combiner_base_path/multi_domain_en_de.laplacian.combiner.ckpt \
     --top_k 16 \
     --kernel laplacian \
     --index_path $database_base_path/multi_domain_en_de_base/trained.index \
     --token_map_path $database_base_path/multi_domain_en_de_base/token_map \
     --embedding_path $database_base_path/multi_domain_en_de_base/embeddings.npy \
     --in_memory False
done

echo "test KSTER multi domain performance in DAMT setting in DE-EN direction"
for domain in wmt14 law medical koran it subtitles
do
echo $domain
python3 -m joeynmt test configs/transformer_base_${domain}_de2en.yaml \
     --ckpt $transformer_base_path/wmt14_de_en.transformer.ckpt \
     --combiner dynamic_combiner \
     --combiner_path $combiner_base_path/multi_domain_de_en.laplacian.combiner.ckpt \
     --top_k 16 \
     --kernel laplacian \
     --index_path $database_base_path/multi_domain_de_en_base/trained.index \
     --token_map_path $database_base_path/multi_domain_de_en_base/token_map \
     --embedding_path $database_base_path/multi_domain_de_en_base/embeddings.npy \
     --in_memory False
done