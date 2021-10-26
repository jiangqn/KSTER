#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

transformer_base_path=checkpoints/transformer
combiner_base_path=checkpoints/combiner
database_base_path=multi_domain_database

echo "test kNN-MT multi domain performance in MDMT setting in EN-DE direction"
for domain in wmt14 law medical koran it subtitles
do
echo $domain
python3 -m joeynmt test configs/transformer_base_${domain}_en2de.yaml \
     --ckpt $transformer_base_path/wmt14_en_de.transformer.ckpt \
     --combiner static_combiner \
     --top_k 16 \
     --mixing_weight 0.7 \
     --bandwidth 10 \
     --kernel gaussian \
     --index_path $database_base_path/multi_domain_en_de_base/trained.index \
     --token_map_path $database_base_path/multi_domain_en_de_base/token_map 
done

echo "test kNN-MT multi domain performance in MDMT setting in DE-EN direction"
for domain in wmt14 law medical koran it subtitles
do
echo $domain
python3 -m joeynmt test configs/transformer_base_${domain}_de2en.yaml \
     --ckpt $transformer_base_path/wmt14_de_en.transformer.ckpt \
     --combiner static_combiner \
     --top_k 16 \
     --mixing_weight 0.9 \
     --bandwidth 10 \
     --kernel gaussian \
     --index_path $database_base_path/multi_domain_de_en_base/trained.index \
     --token_map_path $database_base_path/multi_domain_de_en_base/token_map 
done