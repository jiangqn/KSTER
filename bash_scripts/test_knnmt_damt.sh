#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

transformer_base_path=checkpoints/transformer
combiner_base_path=checkpoints/combiner
database_base_path=database

declare -A en_de_mixing_weight_dict
en_de_mixing_weight_dict=(["law"]=0.8 ["medical"]=0.8 ["koran"]=0.7 ["it"]=0.9 ["subtitles"]=0.7)

declare -A en_de_bandwidth_dict
en_de_bandwidth_dict=(["law"]=10 ["medical"]=10 ["koran"]=10 ["it"]=10 ["subtitles"]=100)

echo "test kNN-MT domain-specific performance in DAMT setting in EN-DE direction"
for domain in law medical koran it subtitles
do
    echo $domain
    mixing_weight=${en_de_mixing_weight_dict[${domain}]}
    bandwidth=${en_de_bandwidth_dict[${domain}]}
    python3 -m joeynmt test configs/transformer_base_${domain}_en2de.yaml \
        --ckpt $transformer_base_path/wmt14_en_de.transformer.ckpt \
        --combiner static_combiner \
        --top_k 16 \
        --mixing_weight $mixing_weight \
        --bandwidth $bandwidth \
        --kernel gaussian \
        --index_path $database_base_path/${domain}_en_de_base/trained.index \
        --token_map_path $database_base_path/${domain}_en_de_base/token_map
done

echo "test kNN-MT general-domain performance in DAMT setting in EN-DE direction"
for domain in law medical koran it subtitles
do
    echo $domain
    mixing_weight=${en_de_mixing_weight_dict[${domain}]}
    bandwidth=${en_de_bandwidth_dict[${domain}]}
    python3 -m joeynmt test configs/transformer_base_wmt14_en2de.yaml \
        --ckpt $transformer_base_path/wmt14_en_de.transformer.ckpt \
        --combiner static_combiner \
        --top_k 16 \
        --mixing_weight $mixing_weight \
        --bandwidth $bandwidth \
        --kernel gaussian \
        --index_path $database_base_path/${domain}_en_de_base/trained.index \
        --token_map_path $database_base_path/${domain}_en_de_base/token_map
done

declare -A de_en_mixing_weight_dict
de_en_mixing_weight_dict=(["law"]=0.7 ["medical"]=0.7 ["koran"]=0.5 ["it"]=0.9 ["subtitles"]=0.6)

declare -A de_en_bandwidth_dict
de_en_bandwidth_dict=(["law"]=10 ["medical"]=10 ["koran"]=10 ["it"]=10 ["subtitles"]=100)

echo "test kNN-MT domain-specific performance in DAMT setting in DE-EN direction"
for domain in law medical koran it subtitles
do
    echo $domain
    mixing_weight=${de_en_mixing_weight_dict[${domain}]}
    bandwidth=${de_en_bandwidth_dict[${domain}]}
    python3 -m joeynmt test configs/transformer_base_${domain}_de2en.yaml \
        --ckpt $transformer_base_path/wmt14_de_en.transformer.ckpt \
        --combiner static_combiner \
        --top_k 16 \
        --mixing_weight $mixing_weight \
        --bandwidth $bandwidth \
        --kernel gaussian \
        --index_path $database_base_path/${domain}_de_en_base/trained.index \
        --token_map_path $database_base_path/${domain}_de_en_base/token_map
done

echo "test kNN-MT domain-specific performance in DAMT setting in DE-EN direction"
for domain in law medical koran it subtitles
do
    echo $domain
    mixing_weight=${de_en_mixing_weight_dict[${domain}]}
    bandwidth=${de_en_bandwidth_dict[${domain}]}
    python3 -m joeynmt test configs/transformer_base_wmt14_de2en.yaml \
        --ckpt $transformer_base_path/wmt14_de_en.transformer.ckpt \
        --combiner static_combiner \
        --top_k 16 \
        --mixing_weight $mixing_weight \
        --bandwidth $bandwidth \
        --kernel gaussian \
        --index_path $database_base_path/${domain}_de_en_base/trained.index \
        --token_map_path $database_base_path/${domain}_de_en_base/token_map
done