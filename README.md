# KSTER

Code for our EMNLP 2021 paper "Learning Kernel-Smoothed Machine Translation with Retrieved Examples" [[paper]](https://arxiv.org/abs/2109.09991).   

## Usage

Download the processed datasets from [this site](https://smailnjueducn-my.sharepoint.com/:f:/g/personal/jiangqn_smail_nju_edu_cn/EoPaArNpYwdJtom_Xy2w2MQBowlwj_vSCyQNfCaQtRcqkg?e=7YToHG). You can also download the built databases from [this site](https://smailnjueducn-my.sharepoint.com/:f:/g/personal/jiangqn_smail_nju_edu_cn/EmcKiw0cc6xNsdi9ceOVhxgBq4yq2lYhlRNLXSBR1Nu1KQ?e=aV9GxX) and download the model checkpoints from [this site](https://smailnjueducn-my.sharepoint.com/:f:/g/personal/jiangqn_smail_nju_edu_cn/EmYMo-Bg-RVLtLJtQ4mqKf4BuMBOOw1odykYWR6CYhRfoQ?e=wBqW9b).

### Train a general-domain base model

Take English -> Germain translation for example.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m joeynmt train configs/transformer_base_wmt14_en2de.yaml
```

### Finetuning trained base model on domain-specific datasets

Take English -> Germain translation in Koran domain for example.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m joeynmt train configs/transformer_base_koran_en2de.yaml
```

### Build database

Take English -> Germain translation in Koran domain for example, wmt14_en_de.transformer.ckpt is the path of trained general-domain base model checkpoint.
```
mkdir database/koran_en_de_base
export CUDA_VISIBLE_DEVICES=0
python3 -m joeynmt build_database configs/transformer_base_koran_en2de.yaml \
        --ckpt wmt14_en_de.transformer.ckpt \
        --division train \
        --index_path database/koran_en_de_base/trained.index \
        --token_map_path database/koran_en_de_base/token_map \
        --embedding_path database/koran_en_de_base/embeddings.npy
```

### Train the bandwidth estimator and weight estimator in KSTER
Take English -> Germain translation in Koran domain for example.
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m joeynmt combiner_train configs/transformer_base_koran_en2de.yaml \
        --ckpt wmt14_en_de.transformer.ckpt \
        --combiner dynamic_combiner \
        --top_k 16 \
        --kernel laplacian \
        --index_path database/koran_en_de_base/trained.index \
        --token_map_path database/koran_en_de_base/token_map \
        --embedding_path database/koran_en_de_base/embeddings.npy \
        --in_memory True
```

### Inference

We unify the inference of base model, finetuned or joint-trained model, kNN-MT and KSTER with a concept of combiner (see joeynmt/combiners.py).

| Combiner type | Methods | Description |
| ---- | ---- | ---- |
| NoCombiner | Base, Finetuning, Joint-training | Directly inference without retrieval. |
| StaticCombiner | kNN-MT | Retrieve similar examples during inference. mixing_weight and bandwidth are pre-specified. |
| DynamicCombiner | KSTER | Retrieve similar examples during inference. mixing_weight and bandwidth are dynamically estimated. |

#### Inference with NoCombiner for Base model

Take English -> Germain translation in Koran domain for example.

```
export CUDA_VISIBLE_DEVICES=0
python3 -m joeynmt test configs/transformer_base_koran_en2de.yaml \
        --ckpt wmt14_en_de.transformer.ckpt \
        --combiner no_combiner
```

#### Inference with StaticCombiner for kNN-MT

Take English -> Germain translation in Koran domain for example.

```
export CUDA_VISIBLE_DEVICES=0
python3 -m joeynmt test configs/transformer_base_koran_en2de.yaml \
        --ckpt wmt14_en_de.transformer.ckpt \
        --combiner static_combiner \
        --top_k 16 \
        --mixing_weight 0.7 \
        --bandwidth 10 \
        --kernel gaussian \
        --index_path database/koran_en_de_base/trained.index \
        --token_map_path database/koran_en_de_base/token_map
```

#### Inference with DynamicCombiner for KSTER

Take English -> Germain translation in Koran domain for example, koran_en_de.laplacian.combiner.ckpt is the path of trained bandwidth estimator and weight estimator for Koran domain.  
--in_memory option specifies whether to load the example embeddings to memory. Set in_memory == True for faster inference, set in_memory == False for lower memory demand.

```
export CUDA_VISIBLE_DEVICES=0
python3 -m joeynmt test configs/transformer_base_koran_en2de.yaml \
        --ckpt wmt14_en_de.transformer.ckpt \
        --combiner dynamic_combiner \
        --combiner_path koran_en_de.laplacian.combiner.ckpt \
        --top_k 16 \
        --kernel laplacian \
        --index_path database/koran_en_de_base/trained.index \
        --token_map_path database/koran_en_de_base/token_map \
        --embedding_path database/koran_en_de_base/embeddings.npy \
        --in_memory True
```

See bash_scripts/test_\*.sh for reproducing our results.  
See logs/\*.log for the logs of our results.

## Acknowledgements

We build the models based on the [joeynmt](https://github.com/joeynmt/joeynmt) codebase.