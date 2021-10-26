# coding: utf-8
"""
This modules holds methods for generating predictions from a model.
"""
import os
import sys
from typing import List, Optional
import logging
import numpy as np
import json

import torch
from torchtext.data import Dataset, Field

from joeynmt.helpers import bpe_postprocess, check_combiner_cfg, load_config, make_logger,\
    get_latest_checkpoint, load_checkpoint, store_attention_plots,\
    get_sacrebleu_description
from joeynmt.metrics import bleu, chrf, token_accuracy, sequence_accuracy
from joeynmt.model import build_model, Model, _DataParallel
from joeynmt.search import run_batch
from joeynmt.batch import Batch
from joeynmt.data import load_data, make_data_iter, MonoDataset
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from joeynmt.vocabulary import Vocabulary
from joeynmt.combiners import build_combiner, NoCombiner

logger = logging.getLogger(__name__)


# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(model: Model, data: Dataset,
                     batch_size: int,
                     use_cuda: bool, max_output_length: int,
                     level: str, eval_metric: Optional[str],
                     n_gpu: int,
                     batch_class: Batch = Batch,
                     compute_loss: bool = False,
                     beam_size: int = 1, beam_alpha: int = -1,
                     batch_type: str = "sentence",
                     postprocess: bool = True,
                     bpe_type: str = "subword-nmt",
                     sacrebleu: dict = None) \
        -> (float, float, float, List[str], List[List[str]], List[str],
            List[str], List[List[str]], List[np.array]):
    """
    Generate translations for the given data.
    If `compute_loss` is True and references are given,
    also compute the loss.

    :param model: model module
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param batch_class: class type of batch
    :param use_cuda: if True, use CUDA
    :param max_output_length: maximum length for generated hypotheses
    :param level: segmentation level, one of "char", "bpe", "word"
    :param eval_metric: evaluation metric, e.g. "bleu"
    :param n_gpu: number of GPUs
    :param compute_loss: whether to computes a scalar loss
        for given inputs and targets
    :param beam_size: beam size for validation.
        If <2 then greedy decoding (default).
    :param beam_alpha: beam search alpha for length penalty,
        disabled if set to -1 (default).
    :param batch_type: validation batch type (sentence or token)
    :param postprocess: if True, remove BPE segmentation from translations
    :param bpe_type: bpe type, one of {"subword-nmt", "sentencepiece"}
    :param sacrebleu: sacrebleu options

    :return:
        - current_valid_score: current validation score [eval_metric],
        - valid_loss: validation loss,
        - valid_ppl:, validation perplexity,
        - valid_sources: validation sources,
        - valid_sources_raw: raw validation sources (before post-processing),
        - valid_references: validation references,
        - valid_hypotheses: validation_hypotheses,
        - decoded_valid: raw validation hypotheses (before post-processing),
        - valid_attention_scores: attention scores for validation hypotheses
    """
    assert batch_size >= n_gpu, "batch_size must be bigger than n_gpu."
    if sacrebleu is None:   # assign default value
        sacrebleu = {"remove_whitespace": True, "tokenize": "13a", "use_detokenization": False}
    if batch_size > 1000 and batch_type == "sentence":
        logger.warning(
            "WARNING: Are you sure you meant to work on huge batches like "
            "this? 'batch_size' is > 1000 for sentence-batching. "
            "Consider decreasing it or switching to"
            " 'eval_batch_type: token'.")
    valid_iter = make_data_iter(
        dataset=data, batch_size=batch_size, batch_type=batch_type,
        shuffle=False, train=False)
    valid_sources_raw = data.src
    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    # disable dropout
    model.eval()
    
    # check combiner
    if not hasattr(model, "combiner"):
        model.combiner = NoCombiner()

    # don't track gradients during validation
    with torch.no_grad():
        all_outputs = []
        valid_attention_scores = []
        total_loss = 0
        total_ntokens = 0
        total_nseqs = 0
        for valid_batch in iter(valid_iter):
            # run as during training to get validation loss (e.g. xent)

            batch = batch_class(valid_batch, pad_index, use_cuda=use_cuda)
            # sort batch now by src length and keep track of order
            sort_reverse_index = batch.sort_by_src_length()

            # run as during training with teacher forcing
            if compute_loss and batch.trg is not None:
                batch_loss, _, _, _ = model(return_type="combiner_loss", **vars(batch))
                if n_gpu > 1:
                    batch_loss = batch_loss.mean() # average on multi-gpu
                total_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            # run as during inference to produce translations
            output, attention_scores = run_batch(
                model=model, batch=batch, beam_size=beam_size,
                beam_alpha=beam_alpha, max_output_length=max_output_length)

            # sort outputs back to original order
            all_outputs.extend(output[sort_reverse_index])
            valid_attention_scores.extend(
                attention_scores[sort_reverse_index]
                if attention_scores is not None else [])

        assert len(all_outputs) == len(data)

        if compute_loss and total_ntokens > 0:
            # total validation loss
            valid_loss = total_loss
            # exponent of token-level negative log prob
            valid_ppl = torch.exp(total_loss / total_ntokens)
        else:
            valid_loss = -1
            valid_ppl = -1

        # decode back to symbols
        decoded_valid = model.trg_vocab.arrays_to_sentences(arrays=all_outputs,
                                                            cut_at_eos=True)

        # evaluate with metric on full dataset
        join_char = " " if level in ["word", "bpe"] else ""
        valid_sources = [join_char.join(s) for s in data.src]
        valid_references = [join_char.join(t) for t in data.trg]
        valid_hypotheses = [join_char.join(t) for t in decoded_valid]

        # post-process
        if level == "bpe" and postprocess:
            valid_sources = [bpe_postprocess(s, bpe_type=bpe_type)
                             for s in valid_sources]
            valid_references = [bpe_postprocess(v, bpe_type=bpe_type)
                                for v in valid_references]
            valid_hypotheses = [bpe_postprocess(v, bpe_type=bpe_type)
                                for v in valid_hypotheses]
        
        if sacrebleu["use_detokenization"]:
            batch_src_detokenize = sacrebleu["batch_src_detokenize"]
            batch_trg_detokenize = sacrebleu["batch_trg_detokenize"]
            valid_sources = batch_src_detokenize(valid_sources)
            valid_references = batch_trg_detokenize(valid_references)
            valid_hypotheses = batch_trg_detokenize(valid_hypotheses)

        # if references are given, evaluate against them
        if valid_references:
            assert len(valid_hypotheses) == len(valid_references)

            current_valid_score = 0
            if eval_metric.lower() == 'bleu':
                # this version does not use any tokenization
                current_valid_score = bleu(
                    valid_hypotheses, valid_references,
                    tokenize=sacrebleu["tokenize"])
            elif eval_metric.lower() == 'chrf':
                current_valid_score = chrf(valid_hypotheses, valid_references,
                    remove_whitespace=sacrebleu["remove_whitespace"])
            elif eval_metric.lower() == 'token_accuracy':
                current_valid_score = token_accuracy(   # supply List[List[str]]
                    list(decoded_valid), list(data.trg))
            elif eval_metric.lower() == 'sequence_accuracy':
                current_valid_score = sequence_accuracy(
                    valid_hypotheses, valid_references)
        else:
            current_valid_score = -1

    return current_valid_score, valid_loss, valid_ppl, valid_sources, \
        valid_sources_raw, valid_references, valid_hypotheses, \
        decoded_valid, valid_attention_scores


def parse_test_args(cfg, mode="test"):
    """
    parse test args
    :param cfg: config object
    :param mode: 'test' or 'translate'
    :return:
    """
    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    batch_size = cfg["training"].get(
        "eval_batch_size", cfg["training"].get("batch_size", 1))
    batch_type = cfg["training"].get(
        "eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = (cfg["training"].get("use_cuda", False)
                and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    if mode == 'test':
        n_gpu = torch.cuda.device_count() if use_cuda else 0
        k = cfg["testing"].get("beam_size", 1)
        batch_per_device = batch_size*k // n_gpu if n_gpu > 1 else batch_size*k
        logger.info("Process device: %s, n_gpu: %d, "
                    "batch_size per device: %d (with beam_size)",
                    device, n_gpu, batch_per_device)
        eval_metric = cfg["training"]["eval_metric"]

    elif mode == 'translate':
        # in multi-gpu, batch_size must be bigger than n_gpu!
        n_gpu = 1 if use_cuda else 0
        logger.debug("Process device: %s, n_gpu: %d", device, n_gpu)
        eval_metric = ""

    level = cfg["data"]["level"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        beam_size = cfg["testing"].get("beam_size", 1)
        beam_alpha = cfg["testing"].get("alpha", -1)
        postprocess = cfg["testing"].get("postprocess", True)
        bpe_type = cfg["testing"].get("bpe_type", "subword-nmt")
        sacrebleu = get_sacrebleu_description(cfg)

    else:
        beam_size = 1
        beam_alpha = -1
        postprocess = True
        bpe_type = "subword-nmt"
        sacrebleu = {"remove_whitespace": True, "tokenize": "13a", "use_detokenization": False}

    decoding_description = "Greedy decoding" if beam_size < 2 else \
        "Beam search decoding with beam size = {} and alpha = {}". \
            format(beam_size, beam_alpha)
    tokenizer_info = f"[{sacrebleu['tokenize']}]" \
        if eval_metric == "bleu" else ""

    return batch_size, batch_type, use_cuda, device, n_gpu, level, \
           eval_metric, max_output_length, beam_size, beam_alpha, \
           postprocess, bpe_type, sacrebleu, decoding_description, \
           tokenizer_info


# pylint: disable-msg=logging-too-many-args
def test(cfg_file,
         ckpt: str,
         combiner_cfg: dict,
         batch_class: Batch = Batch,
         output_path: str = None,
         save_attention: bool = False,
         datasets: dict = None) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param batch_class: class type of batch
    :param output_path: path to output
    :param datasets: datasets to predict
    :param save_attention: whether to save the computed attention weights
    """

    cfg = load_config(cfg_file)
    model_dir = cfg["training"]["model_dir"]

    check_combiner_cfg(combiner_cfg)
    cfg["combiner"] = combiner_cfg

    if len(logger.handlers) == 0:
        _ = make_logger(model_dir, mode="test")   # version string returned

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir)
        try:
            step = ckpt.split(model_dir+"/")[1].split(".ckpt")[0]
        except IndexError:
            step = "best"

    # load the data
    if datasets is None:
        _, dev_data, test_data, src_vocab, trg_vocab = load_data(
            data_cfg=cfg["data"], datasets=["dev", "test"])
        data_to_predict = {"dev": dev_data, "test": test_data}
    else:  # avoid to load data again
        data_to_predict = {"dev": datasets["dev"], "test": datasets["test"]}
        src_vocab = datasets["src_vocab"]
        trg_vocab = datasets["trg_vocab"]

    # parse test args
    batch_size, batch_type, use_cuda, device, n_gpu, level, eval_metric, \
        max_output_length, beam_size, beam_alpha, postprocess, \
        bpe_type, sacrebleu, decoding_description, tokenizer_info \
        = parse_test_args(cfg, mode="test")

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    combiner = build_combiner(cfg)
    # load combiner from checkpoint for dynamic combiners
    if combiner_cfg["type"]=="dynamic_combiner":
        combiner_checkpoint = load_checkpoint(combiner_cfg["combiner_path"], use_cuda=use_cuda)
        combiner.load_state_dict(combiner_checkpoint["model_state"])

    model.combiner = combiner

    if use_cuda:
        model.to(device)

    # multi-gpu eval
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = _DataParallel(model)

    for data_set_name, data_set in data_to_predict.items():
        if data_set is None:
            continue

        dataset_file = cfg["data"][data_set_name] + "." + cfg["data"]["trg"]
        logger.info("Decoding on %s set (%s)...", data_set_name, dataset_file)

        #pylint: disable=unused-variable
        score, loss, ppl, sources, sources_raw, references, hypotheses, \
        hypotheses_raw, attention_scores = validate_on_data(
            model, data=data_set, batch_size=batch_size,
            batch_class=batch_class, batch_type=batch_type, level=level,
            max_output_length=max_output_length, eval_metric=eval_metric,
            use_cuda=use_cuda, compute_loss=False, beam_size=beam_size,
            beam_alpha=beam_alpha, postprocess=postprocess,
            bpe_type=bpe_type, sacrebleu=sacrebleu, n_gpu=n_gpu)
        #pylint: enable=unused-variable

        if "trg" in data_set.fields:
            logger.info("%4s %s%s: %6.2f [%s]",
                        data_set_name, eval_metric, tokenizer_info,
                        score, decoding_description)
        else:
            logger.info("No references given for %s -> no evaluation.",
                        data_set_name)

        if save_attention:
            if attention_scores:
                attention_name = "{}.{}.att".format(data_set_name, step)
                attention_path = os.path.join(model_dir, attention_name)
                logger.info("Saving attention plots. This might take a while..")
                store_attention_plots(attentions=attention_scores,
                                      targets=hypotheses_raw,
                                      sources=data_set.src,
                                      indices=range(len(hypotheses)),
                                      output_prefix=attention_path)
                logger.info("Attention plots saved to: %s", attention_path)
            else:
                logger.warning("Attention scores could not be saved. "
                               "Note that attention scores are not available "
                               "when using beam search. "
                               "Set beam_size to 1 for greedy decoding.")

        if output_path is not None:
            output_path_set = "{}.{}".format(output_path, data_set_name)
            with open(output_path_set, mode="w", encoding="utf-8") as out_file:
                for hyp in hypotheses:
                    out_file.write(hyp + "\n")
            logger.info("Translations saved to: %s", output_path_set)

# pylint: disable-msg=logging-too-many-args
def analyze(cfg_file,
         ckpt: str,
         combiner_cfg: dict,
         output_path: str,
         batch_class: Batch = Batch,
         datasets: dict = None) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param batch_class: class type of batch
    :param output_path: path to output
    :param datasets: datasets to predict
    :param save_attention: whether to save the computed attention weights
    """

    cfg = load_config(cfg_file)
    model_dir = cfg["training"]["model_dir"]

    check_combiner_cfg(combiner_cfg)
    cfg["combiner"] = combiner_cfg

    if len(logger.handlers) == 0:
        _ = make_logger(model_dir, mode="test")   # version string returned

    # load the data
    if datasets is None:
        _, dev_data, test_data, src_vocab, trg_vocab = load_data(
            data_cfg=cfg["data"], datasets=["dev", "test"])
        data_to_predict = {"dev": dev_data, "test": test_data}
    else:  # avoid to load data again
        data_to_predict = {"dev": datasets["dev"], "test": datasets["test"]}
        src_vocab = datasets["src_vocab"]
        trg_vocab = datasets["trg_vocab"]

    # parse test args
    batch_size, batch_type, use_cuda, device, n_gpu, level, eval_metric, \
        max_output_length, beam_size, beam_alpha, postprocess, \
        bpe_type, sacrebleu, decoding_description, tokenizer_info \
        = parse_test_args(cfg, mode="test")

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    combiner = build_combiner(cfg)
    # load combiner from checkpoint for dynamic combiners
    if combiner_cfg["type"]=="dynamic_combiner":
        combiner_checkpoint = load_checkpoint(combiner_cfg["combiner_path"], use_cuda=use_cuda)
        combiner.load_state_dict(combiner_checkpoint["model_state"])

    model.combiner = combiner

    if use_cuda:
        model.to(device)

    # multi-gpu eval
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = _DataParallel(model)

    for data_set_name, data_set in data_to_predict.items():
        if data_set is None:
            continue

        dataset_file = cfg["data"][data_set_name] + "." + cfg["data"]["trg"]
        logger.info("Force decoding on %s set (%s)...", data_set_name, dataset_file)

        valid_iter = make_data_iter(
        dataset=data_set, batch_size=batch_size, batch_type=batch_type,
        shuffle=False, train=False)

        valid_sources_raw = data_set.src
        pad_index = model.src_vocab.stoi[PAD_TOKEN]
        # disable dropout
        model.eval()
        # don't track gradients during validation
        all_docs = []
        with torch.no_grad():
            for step, valid_batch in enumerate(iter(valid_iter)):
                # run as during training to get validation loss (e.g. xent)

                batch = batch_class(valid_batch, pad_index, use_cuda=use_cuda)
                trg_length = batch.trg_length.cpu().numpy().tolist()
                trg = batch.trg
                batch_src_texts = model.src_vocab.arrays_to_sentences(arrays=batch.src, cut_at_eos=True)
                batch_trg_texts = model.trg_vocab.arrays_to_sentences(arrays=batch.trg, cut_at_eos=False)

                # sort batch now by src length and keep track of order
                sort_reverse_index = batch.sort_by_src_length()

                batch_logits, batch_hidden_states, _, _ = model._encode_decode(**vars(batch))
                batch_logits = batch_logits[sort_reverse_index]
                batch_hidden_states = batch_hidden_states[sort_reverse_index]

                batch_mixed_distribution, batch_model_based_distribution, example_based_distribution, batch_mixing_weight, batch_bandwidth = model.combiner.detailed_forward(batch_hidden_states, batch_logits)
                batch_preds, batch_model_preds, batch_example_preds = batch_mixed_distribution.argmax(dim=-1), batch_model_based_distribution.argmax(dim=-1), example_based_distribution.argmax(dim=-1)
                batch_trg_probs = torch.gather(batch_mixed_distribution, -1, trg.unsqueeze(-1)).squeeze(-1)
                batch_trg_model_based_distribution = torch.gather(batch_model_based_distribution, -1, trg.unsqueeze(-1)).squeeze(-1)
                batch_trg_example_based_distribution = torch.gather(example_based_distribution, -1, trg.unsqueeze(-1)).squeeze(-1)

                batch_preds, batch_model_preds, batch_example_preds = batch_preds.cpu().numpy(), batch_model_preds.cpu().numpy(), batch_example_preds.cpu().numpy()
                batch_trg_probs, batch_trg_model_based_distribution, batch_trg_example_based_distribution = batch_trg_probs.cpu().numpy(), batch_trg_model_based_distribution.cpu().numpy(), batch_trg_example_based_distribution.cpu().numpy()
                batch_mixing_weight, batch_bandwidth = batch_mixing_weight.cpu().numpy(), batch_bandwidth.cpu().numpy()

                for i in range(len(batch_trg_texts)):
                    trg_len = trg_length[i] - 1
                    doc = {
                        "src": " ".join(batch_src_texts[i]),
                        "trg_tokens": batch_trg_texts[i][0: trg_len],
                        "preds": batch_preds[i][0: trg_len].tolist(),
                        "model_preds": batch_model_preds[i][0: trg_len].tolist(),
                        "example_preds": batch_example_preds[i][0: trg_len].tolist(),
                        "trg_probs": batch_trg_probs[i][0: trg_len].tolist(),
                        "trg_model_probs": batch_trg_model_based_distribution[i][0: trg_len].tolist(),
                        "trg_example_probs": batch_trg_example_based_distribution[i][0: trg_len].tolist(),
                        "mixing_weight": batch_mixing_weight[i][0: trg_len].tolist(),
                        "bandwidth": batch_bandwidth[i][0: trg_len].tolist()
                    }
                    all_docs.append(json.dumps(doc))

        output_path_set = "{}.{}".format(output_path, data_set_name)
        with open(output_path_set, mode="w", encoding="utf-8") as out_file:
            for doc in all_docs:
                out_file.write(doc + "\n")
        logger.info("Analysis saved to: %s", output_path_set)

# pylint: disable-msg=logging-too-many-args
def score_translations(cfg_file,
         ckpt: str,
         combiner_cfg: dict,
         output_path: str,
         batch_class: Batch = Batch,
         datasets: dict = None) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param batch_class: class type of batch
    :param output_path: path to output
    :param datasets: datasets to predict
    :param save_attention: whether to save the computed attention weights
    """

    from joeynmt.loss import XentLoss
    from joeynmt.constants import PAD_TOKEN

    cfg = load_config(cfg_file)
    model_dir = cfg["training"]["model_dir"]

    check_combiner_cfg(combiner_cfg)
    cfg["combiner"] = combiner_cfg

    if len(logger.handlers) == 0:
        _ = make_logger(model_dir, mode="test")   # version string returned

    # load the data
    if datasets is None:
        _, dev_data, test_data, src_vocab, trg_vocab = load_data(
            data_cfg=cfg["data"], datasets=["dev", "test"])
        data_to_predict = {"dev": dev_data, "test": test_data}
    else:  # avoid to load data again
        data_to_predict = {"dev": datasets["dev"], "test": datasets["test"]}
        src_vocab = datasets["src_vocab"]
        trg_vocab = datasets["trg_vocab"]

    # parse test args
    batch_size, batch_type, use_cuda, device, n_gpu, level, eval_metric, \
        max_output_length, beam_size, beam_alpha, postprocess, \
        bpe_type, sacrebleu, decoding_description, tokenizer_info \
        = parse_test_args(cfg, mode="test")

    batch_size = 1
    batch_type = "sentence"

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    combiner = build_combiner(cfg)
    # load combiner from checkpoint for dynamic combiners
    if combiner_cfg["type"]=="dynamic_combiner":
        combiner_checkpoint = load_checkpoint(combiner_cfg["combiner_path"], use_cuda=use_cuda)
        combiner.load_state_dict(combiner_checkpoint["model_state"])

    model.combiner = combiner
    loss_function = XentLoss(pad_index=model.pad_index, smoothing=0.1)

    if use_cuda:
        model.to(device)

    # multi-gpu eval
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = _DataParallel(model)

    for data_set_name, data_set in data_to_predict.items():
        if data_set is None:
            continue
        if data_set_name == "dev":
            continue

        dataset_file = cfg["data"][data_set_name] + "." + cfg["data"]["trg"]
        logger.info("Force decoding on %s set (%s)...", data_set_name, dataset_file)

        valid_iter = make_data_iter(
        dataset=data_set, batch_size=batch_size, batch_type=batch_type,
        shuffle=False, train=False)

        valid_sources_raw = data_set.src
        pad_index = model.src_vocab.stoi[PAD_TOKEN]
        # disable dropout
        model.eval()
        # don't track gradients during validation
        losses = []
        with torch.no_grad():
            for step, valid_batch in enumerate(iter(valid_iter)):
                # run as during training to get validation loss (e.g. xent)

                batch = batch_class(valid_batch, pad_index, use_cuda=use_cuda)
                trg_length = batch.trg_length.cpu().numpy().tolist()
                trg = batch.trg

                # sort batch now by src length and keep track of order
                sort_reverse_index = batch.sort_by_src_length()

                batch_logits, batch_hidden_states, _, _ = model._encode_decode(**vars(batch))
                batch_logits = batch_logits[sort_reverse_index]
                batch_hidden_states = batch_hidden_states[sort_reverse_index]

                batch_log_probs = model.combiner(batch_hidden_states, batch_logits)
                loss = loss_function(batch_log_probs, trg).item()
                losses.append(loss)
        
        if output_path != None:
            with open(output_path, mode="w", encoding="utf-8") as out_file:
                for loss in losses:
                    out_file.write(f"{loss}\n")
            logger.info(f"Losses saved to: {output_path}")

def translate(cfg_file: str,
              ckpt: str,
              output_path: str = None,
              batch_class: Batch = Batch) -> None:
    """
    Interactive translation function.
    Loads model from checkpoint and translates either the stdin input or
    asks for input to translate interactively.
    The input has to be pre-processed according to the data that the model
    was trained on, i.e. tokenized or split into subwords.
    Translations are printed to stdout.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output file
    :param batch_class: class type of batch
    """

    def _load_line_as_data(line):
        """ Create a dataset from one line via a temporary file. """
        # write src input to temporary file
        tmp_name = "tmp"
        tmp_suffix = ".src"
        tmp_filename = tmp_name+tmp_suffix
        with open(tmp_filename, "w") as tmp_file:
            tmp_file.write("{}\n".format(line))

        test_data = MonoDataset(path=tmp_name, ext=tmp_suffix,
                                field=src_field)

        # remove temporary file
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

        return test_data

    def _translate_data(test_data):
        """ Translates given dataset, using parameters from outer scope. """
        # pylint: disable=unused-variable
        score, loss, ppl, sources, sources_raw, references, hypotheses, \
        hypotheses_raw, attention_scores = validate_on_data(
            model, data=test_data, batch_size=batch_size,
            batch_class=batch_class, batch_type=batch_type, level=level,
            max_output_length=max_output_length, eval_metric="",
            use_cuda=use_cuda, compute_loss=False, beam_size=beam_size,
            beam_alpha=beam_alpha, postprocess=postprocess,
            bpe_type=bpe_type, sacrebleu=sacrebleu, n_gpu=n_gpu)
        return hypotheses

    cfg = load_config(cfg_file)
    model_dir = cfg["training"]["model_dir"]

    _ = make_logger(model_dir, mode="translate")
    # version string returned

    # when checkpoint is not specified, take oldest from model dir
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir)

    # read vocabs
    src_vocab_file = cfg["data"].get("src_vocab", model_dir + "/src_vocab.txt")
    trg_vocab_file = cfg["data"].get("trg_vocab", model_dir + "/trg_vocab.txt")
    src_vocab = Vocabulary(file=src_vocab_file)
    trg_vocab = Vocabulary(file=trg_vocab_file)

    data_cfg = cfg["data"]
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = Field(init_token=None, eos_token=EOS_TOKEN,
                      pad_token=PAD_TOKEN, tokenize=tok_fun,
                      batch_first=True, lower=lowercase,
                      unk_token=UNK_TOKEN,
                      include_lengths=True)
    src_field.vocab = src_vocab

    # parse test args
    batch_size, batch_type, use_cuda, device, n_gpu, level, _, \
        max_output_length, beam_size, beam_alpha, postprocess, \
        bpe_type, sacrebleu, _, _ = parse_test_args(cfg, mode="translate")

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.to(device)

    if not sys.stdin.isatty():
        # input file given
        test_data = MonoDataset(path=sys.stdin, ext="", field=src_field)
        hypotheses = _translate_data(test_data)

        if output_path is not None:
            # write to outputfile if given
            output_path_set = "{}".format(output_path)
            with open(output_path_set, mode="w", encoding="utf-8") as out_file:
                for hyp in hypotheses:
                    out_file.write(hyp + "\n")
            logger.info("Translations saved to: %s.", output_path_set)
        else:
            # print to stdout
            for hyp in hypotheses:
                print(hyp)

    else:
        # enter interactive mode
        batch_size = 1
        batch_type = "sentence"
        while True:
            try:
                src_input = input("\nPlease enter a source sentence "
                                  "(pre-processed): \n")
                if not src_input.strip():
                    break

                # every line has to be made into dataset
                test_data = _load_line_as_data(line=src_input)

                hypotheses = _translate_data(test_data)
                print("JoeyNMT: {}".format(hypotheses[0]))

            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
