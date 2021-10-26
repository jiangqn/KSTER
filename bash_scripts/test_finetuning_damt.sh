#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

transformer_base_path=checkpoints/transformer
combiner_base_path=checkpoints/combiner
database_base_path=database

echo "test finetuned model domain-specific performance in DAMT setting in EN-DE direction"
for domain in law medical koran it subtitles
do
echo $domain
python3 -m joeynmt test configs/transformer_base_${domain}_en2de.yaml \
     --ckpt $transformer_base_path/${domain}_en_de.finetune.transformer.ckpt \
     --combiner no_combiner
done

echo "test finetuned model general-domain performance in DAMT setting in EN-DE direction"
for domain in law medical koran it subtitles
do
echo $domain
python3 -m joeynmt test configs/transformer_base_wmt14_en2de.yaml \
     --ckpt $transformer_base_path/${domain}_en_de.finetune.transformer.ckpt \
     --combiner no_combiner
done

echo "test finetuned model domain-specific performance in DAMT setting in DE-EN direction"
for domain in law medical koran it subtitles
do
echo $domain
python3 -m joeynmt test configs/transformer_base_${domain}_de2en.yaml \
     --ckpt $transformer_base_path/${domain}_de_en.finetune.transformer.ckpt \
     --combiner no_combiner
done

echo "test finetuned model general-domain performance in DAMT setting in DE-EN direction"
for domain in law medical koran it subtitles
do
echo $domain
python3 -m joeynmt test configs/transformer_base_wmt14_de2en.yaml \
     --ckpt $transformer_base_path/${domain}_de_en.finetune.transformer.ckpt \
     --combiner no_combiner
done