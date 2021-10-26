#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

transformer_base_path=checkpoints/transformer
combiner_base_path=checkpoints/combiner
database_base_path=database

echo "test KSTER domain-specific performance in DAMT setting in EN-DE direction"
for domain in law medical koran it subtitles
do
echo $domain
python3 -m joeynmt test configs/transformer_base_${domain}_en2de.yaml \
     --ckpt $transformer_base_path/wmt14_en_de.transformer.ckpt \
     --combiner dynamic_combiner \
     --combiner_path $combiner_base_path/${domain}_en_de.laplacian.combiner.ckpt \
     --top_k 16 \
     --kernel laplacian \
     --index_path $database_base_path/${domain}_en_de_base/trained.index \
     --token_map_path $database_base_path/${domain}_en_de_base/token_map \
     --embedding_path $database_base_path/${domain}_en_de_base/embeddings.npy \
     --in_memory False
done

echo "test KSTER general-domain performance in DAMT setting in EN-DE direction"
for domain in law medical koran it subtitles
do
echo $domain
python3 -m joeynmt test configs/transformer_base_wmt14_en2de.yaml \
     --ckpt $transformer_base_path/wmt14_en_de.transformer.ckpt \
     --combiner dynamic_combiner \
     --combiner_path $combiner_base_path/${domain}_en_de.laplacian.combiner.ckpt \
     --top_k 16 \
     --kernel laplacian \
     --index_path $database_base_path/${domain}_en_de_base/trained.index \
     --token_map_path $database_base_path/${domain}_en_de_base/token_map \
     --embedding_path $database_base_path/${domain}_en_de_base/embeddings.npy \
     --in_memory False
done

echo "test KSTER domain-specific performance in DAMT setting in DE-EN direction"
for domain in law medical koran it subtitles
do
echo $domain
python3 -m joeynmt test configs/transformer_base_${domain}_de2en.yaml \
     --ckpt $transformer_base_path/wmt14_de_en.transformer.ckpt \
     --combiner dynamic_combiner \
     --combiner_path $combiner_base_path/${domain}_de_en.laplacian.combiner.ckpt \
     --top_k 16 \
     --kernel laplacian \
     --index_path $database_base_path/${domain}_de_en_base/trained.index \
     --token_map_path $database_base_path/${domain}_de_en_base/token_map \
     --embedding_path $database_base_path/${domain}_de_en_base/embeddings.npy \
     --in_memory False
done

echo "test KSTER general-domain performance in DAMT setting in DE-EN direction"
for domain in law medical koran it subtitles
do
echo $domain
python3 -m joeynmt test configs/transformer_base_wmt14_de2en.yaml \
     --ckpt $transformer_base_path/wmt14_de_en.transformer.ckpt \
     --combiner dynamic_combiner \
     --combiner_path $combiner_base_path/${domain}_de_en.laplacian.combiner.ckpt \
     --top_k 16 \
     --kernel laplacian \
     --index_path $database_base_path/${domain}_de_en_base/trained.index \
     --token_map_path $database_base_path/${domain}_de_en_base/token_map \
     --embedding_path $database_base_path/${domain}_de_en_base/embeddings.npy \
     --in_memory False
done