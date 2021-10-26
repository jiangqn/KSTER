#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

transformer_base_path=checkpoints/transformer
combiner_base_path=checkpoints/combiner
database_base_path=database

echo "test base model domain-specific performance in DAMT setting in EN-DE direction"
for domain in law medical koran it subtitles
do
echo $domain
python3 -m joeynmt test configs/transformer_base_${domain}_en2de.yaml \
     --ckpt $transformer_base_path/wmt14_en_de.transformer.ckpt \
     --combiner no_combiner
done

echo "test base model general-domain performance in DAMT setting in EN-DE direction"
python3 -m joeynmt test configs/transformer_base_wmt14_en2de.yaml \
     --ckpt $transformer_base_path/wmt14_en_de.transformer.ckpt \
     --combiner no_combiner

echo "test base model domain-specific performance in DAMT setting in DE-EN direction"
for domain in law medical koran it subtitles
do
echo $domain
python3 -m joeynmt test configs/transformer_base_${domain}_de2en.yaml \
     --ckpt $transformer_base_path/wmt14_de_en.transformer.ckpt \
     --combiner no_combiner
done

echo "test base model general-domain performance in DAMT setting in DE-EN direction"
python3 -m joeynmt test configs/transformer_base_wmt14_de2en.yaml \
     --ckpt $transformer_base_path/wmt14_de_en.transformer.ckpt \
     --combiner no_combiner

echo "finish"