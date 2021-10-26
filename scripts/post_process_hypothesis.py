from joeynmt.vocabulary import Vocabulary
import os
import subprocess
import yaml
import glob
from sacremoses import MosesTokenizer, MosesDetokenizer
import spacy
from collections import Counter

config_path = glob.glob("*.yaml")[0]
config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

src_lang = config["data"]["src"]
trg_lang = config["data"]["trg"]

print(f"src_lang: {src_lang}\ttrg_lang: {trg_lang}")

base_path = "analysis"

detokenized_base_path = os.path.join(base_path, "detokenized")
tokenized_base_path = os.path.join(base_path, "tokenized")
bpe_base_path = os.path.join(base_path, "bpe")

detokenized_dev_path = os.path.join(detokenized_base_path, f"dev.{trg_lang}")
detokenized_test_path = os.path.join(detokenized_base_path, f"test.{trg_lang}")

tokenized_dev_path = os.path.join(tokenized_base_path, f"dev.tok.{trg_lang}")
tokenized_test_path = os.path.join(tokenized_base_path, f"test.tok.{trg_lang}")

bpe_dev_path = os.path.join(bpe_base_path, f"dev.bpe.32k.{trg_lang}")
bpe_test_path = os.path.join(bpe_base_path, f"test.bpe.32k.{trg_lang}")

print("rename hypothesis files")
subprocess.call(f"mv {base_path}/beam4_alpha0.6.dev {base_path}/dev.{trg_lang}", shell=True)
subprocess.call(f"mv {base_path}/beam4_alpha0.6.test {base_path}/test.{trg_lang}", shell=True)

if not os.path.exists(detokenized_base_path):
    os.makedirs(detokenized_base_path)

print(f"copy hypothesis files into {detokenized_base_path}")
subprocess.call(f"cp {base_path}/dev.{trg_lang} {detokenized_base_path}", shell=True)
subprocess.call(f"cp {base_path}/test.{trg_lang} {detokenized_base_path}", shell=True)

def file_tokenize(src_path: str, trg_path: str, lang: str) -> None:
    src_file = open(src_path, "r", encoding="utf-8")
    trg_file = open(trg_path, "w", encoding="utf-8")
    tokenizer = MosesTokenizer(lang=lang)
    for line in src_file.readlines():
        line = " ".join(tokenizer.tokenize(line.strip(), aggressive_dash_splits=True, escape=False))
        trg_file.write(line + "\n")
    src_file.close()
    trg_file.close()
        
if not os.path.exists(tokenized_base_path):
    os.makedirs(tokenized_base_path)

print("tokenize hypothesis")
file_tokenize(detokenized_dev_path, tokenized_dev_path, trg_lang)
file_tokenize(detokenized_test_path, tokenized_test_path, trg_lang)

if not os.path.exists(bpe_base_path):
    os.makedirs(bpe_base_path)

dataset_base_path = os.path.dirname(config["data"]["train"])
codes_path = os.path.join(dataset_base_path, "codes.txt")
vocabulary_path = os.path.join(dataset_base_path, f"vocabulary.{trg_lang}")

print("segment word into subwords with bpe")
subprocess.call(f"subword-nmt apply-bpe -c {codes_path} --vocabulary {vocabulary_path} --vocabulary-threshold 50 < {tokenized_dev_path} > {bpe_dev_path}", shell=True)
subprocess.call(f"subword-nmt apply-bpe -c {codes_path} --vocabulary {vocabulary_path} --vocabulary-threshold 50 < {tokenized_test_path} > {bpe_test_path}", shell=True)

print(f"copy src files to {bpe_base_path}")
subprocess.call("cat %s >> %s" % (config["data"]["dev"] + "." + src_lang, f"{bpe_base_path}/dev.bpe.32k.{src_lang}"), shell=True)
subprocess.call("cat %s >> %s" % (config["data"]["test"] + "." + src_lang, f"{bpe_base_path}/test.bpe.32k.{src_lang}"), shell=True)
subprocess.call("cp %s %s" % (os.path.join(dataset_base_path, "vocab.txt"), bpe_base_path), shell=True)

model_dict = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm"
}

#subprocess.call("python3 -m spacy download %s" % model_dict[trg_lang], shell=True)

nlp = spacy.load(model_dict[trg_lang])

def compute_pos_tag_for_tokenized_file(src_path: str, trg_path: str) -> None:
    src_file = open(src_path, "r", encoding="utf-8")
    trg_file = open(trg_path, "w", encoding="utf-8")
    for line in src_file.readlines():
        line = line.strip()
        words = line.split()
        spaces = [True for _ in range(len(words) - 1)] + [False]
        doc = spacy.tokens.doc.Doc(nlp.vocab, words=words, spaces=spaces)
        for name, proc in nlp.pipeline:
            doc = proc(doc)
        pos_tags = []
        for token in doc:
            pos_tags.append(str(token.pos_))
        pos_tags_line = " ".join(pos_tags)
        trg_file.write(pos_tags_line + "\n")
    src_file.close()
    trg_file.close()

print("compute pos tag for tokenized file")
compute_pos_tag_for_tokenized_file(tokenized_dev_path, os.path.join(tokenized_base_path, f"dev.{trg_lang}.pos"))
compute_pos_tag_for_tokenized_file(tokenized_test_path, os.path.join(tokenized_base_path, f"test.{trg_lang}.pos"))

def assign_pos_tag_for_bpe(src_path: str, bpe_path: str, trg_path: str) -> None:
    src_file = open(src_path, "r", encoding="utf-8")
    bpe_file = open(bpe_path, "r", encoding="utf-8")
    trg_file = open(trg_path, "w", encoding="utf-8")
    for src_pos_line, bpe_line in zip(src_file.readlines(), bpe_file.readlines()):
        src_pos_line = src_pos_line.strip().split()
        bpe_line = bpe_line.strip().split()
        trg_pos_tags = []
        p = 0
        for subword in bpe_line:
            trg_pos_tags.append(src_pos_line[p])
            if not subword.endswith("@@"):
                p += 1
        assert p == len(src_pos_line)
        assert len(bpe_line) == len(trg_pos_tags)
        trg_pos_tags_line = " ".join(trg_pos_tags)
        trg_file.write(trg_pos_tags_line + "\n")

    src_file.close()
    bpe_file.close()
    trg_file.close()

print("assign pos tag for bpe")
assign_pos_tag_for_bpe(os.path.join(tokenized_base_path, f"dev.{trg_lang}.pos"), os.path.join(bpe_base_path, f"dev.bpe.32k.{trg_lang}"),
    os.path.join(bpe_base_path, f"dev.bpe.32k.{trg_lang}.pos"))
assign_pos_tag_for_bpe(os.path.join(tokenized_base_path, f"test.{trg_lang}.pos"), os.path.join(bpe_base_path, f"test.bpe.32k.{trg_lang}"),
    os.path.join(bpe_base_path, f"test.bpe.32k.{trg_lang}.pos"))

token_file = open("token_map", "r", encoding="utf-8")
token_map = [int(x.strip()) for x in token_file.readlines()]
token_file.close()
counter = Counter(token_map)

vocab_file = open(config["data"]["trg_vocab"], "r", encoding="utf-8")
vocab_size = len(vocab_file.readlines()) + 4
vocab_file.close()

frequency = [counter[i] if i in counter else 0 for i in range(vocab_size)]
frequency_file = open(os.path.join(base_path, "token_frequency.txt"), "w", encoding="utf-8")
for c in frequency:
    frequency_file.write(str(c) + "\n")
frequency_file.close()

config["data"]["dev"] = os.path.join(bpe_base_path, "dev.bpe.32k")
config["data"]["test"] = os.path.join(bpe_base_path, "test.bpe.32k")

with open("analysis_" + config_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f)