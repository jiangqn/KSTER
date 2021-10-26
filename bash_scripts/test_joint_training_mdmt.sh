#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

transformer_base_path=checkpoints/transformer
combiner_base_path=checkpoints/combiner
database_base_path=multi_domain_database

echo "test joint-training multi-domain in MDMT setting in EN-DE direction"
for domain in wmt14 law medical koran it subtitles
do
echo $domain
python3 -m joeynmt test configs/transformer_base_${domain}_en2de.yaml \
     --ckpt $transformer_base_path/multi_domain_en_de.transformer.ckpt \
     --combiner no_combiner
done

echo "test joint-training multi-domain performance in MDMT setting in DE-EN direction"
for domain in wmt14 law medical koran it subtitles
do
echo $domain
python3 -m joeynmt test configs/transformer_base_${domain}_de2en.yaml \
     --ckpt $transformer_base_path/multi_domain_de_en.transformer.ckpt \
     --combiner no_combiner
done