# coding: utf-8
"""
Training module
"""

import argparse
import time
import shutil
from typing import List
import logging
import os
import sys
import collections
import pathlib
import numpy as np

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from torchtext.data import Dataset

from joeynmt.model import build_model
from joeynmt.batch import Batch
from joeynmt.helpers import log_data_info, load_config, log_cfg, \
    store_attention_plots, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, latest_checkpoint_update, \
    ConfigurationError, get_sacrebleu_description, check_combiner_cfg
from joeynmt.model import Model, _DataParallel
from joeynmt.prediction import validate_on_data
from joeynmt.loss import XentLoss
from joeynmt.data import load_data, make_data_iter
from joeynmt.builders import build_optimizer, build_scheduler, \
    build_gradient_clipper
from joeynmt.prediction import test
from joeynmt.combiners import build_combiner

# for fp16 training
try:
    from apex import amp
    amp.register_half_function(torch, "einsum")
except ImportError as no_apex:
    # error handling in CombinerTrainManager object construction
    pass

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
class CombinerTrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""

    def __init__(self, model: Model, config: dict,
                 batch_class: Batch = Batch) -> None:
        """
        Creates a new CombinerTrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        :param batch_class: batch class to encapsulate the torch class
        """
        combiner_train_config = config["combiner_training"]
        self.batch_class = batch_class

        # files for logging and storing
        self.model_dir = combiner_train_config["model_dir"]
        assert os.path.exists(self.model_dir)

        self.logging_freq = combiner_train_config.get("logging_freq", 100)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=self.model_dir + "/tensorboard/")

        self.save_latest_checkpoint = combiner_train_config.get("save_latest_ckpt", True)

        # model
        self.model = model
        self._log_parameters_list()

        # objective
        self.label_smoothing = combiner_train_config.get("label_smoothing", 0.0)
        self.model.loss_function = XentLoss(pad_index=self.model.pad_index,
                                            smoothing=self.label_smoothing)
        self.normalization = combiner_train_config.get("normalization", "batch")
        if self.normalization not in ["batch", "tokens", "none"]:
            raise ConfigurationError("Invalid normalization option."
                                     "Valid options: "
                                     "'batch', 'tokens', 'none'.")

        # optimization
        self.learning_rate_min = combiner_train_config.get("learning_rate_min", 1.0e-8)

        self.clip_grad_fun = build_gradient_clipper(config=combiner_train_config)
        self.optimizer = build_optimizer(config=combiner_train_config,
                                         parameters=self.model.combiner.parameters())

        # validation & early stopping
        self.validation_freq = combiner_train_config.get("validation_freq", 1000)
        self.log_valid_sents = combiner_train_config.get("print_valid_sents", [0, 1, 2])
        self.ckpt_queue = collections.deque(
            maxlen=combiner_train_config.get("keep_last_ckpts", 5))
        self.eval_metric = combiner_train_config.get("eval_metric", "bleu")
        if self.eval_metric not in [
                'bleu', 'chrf', 'token_accuracy', 'sequence_accuracy'
        ]:
            raise ConfigurationError("Invalid setting for 'eval_metric', "
                                     "valid options: 'bleu', 'chrf', "
                                     "'token_accuracy', 'sequence_accuracy'.")
        self.early_stopping_metric = combiner_train_config.get("early_stopping_metric",
                                                      "eval_metric")

        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for this metric.
        # If we schedule after BLEU/chrf/accuracy, we want to maximize the
        # score, else we want to minimize it.
        if self.early_stopping_metric in ["ppl", "loss"]:
            self.minimize_metric = True
        elif self.early_stopping_metric == "eval_metric":
            if self.eval_metric in [
                    "bleu", "chrf", "token_accuracy", "sequence_accuracy"
            ]:
                self.minimize_metric = False
            # eval metric that has to get minimized (not yet implemented)
            else:
                self.minimize_metric = True
        else:
            raise ConfigurationError(
                "Invalid setting for 'early_stopping_metric', "
                "valid options: 'loss', 'ppl', 'eval_metric'.")

        # eval options
        test_config = config["testing"]
        self.bpe_type = test_config.get("bpe_type", "subword-nmt")
        self.sacrebleu = get_sacrebleu_description(config)

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=combiner_train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"])

        # data & batch handling
        self.level = config["data"]["level"]
        if self.level not in ["word", "bpe", "char"]:
            raise ConfigurationError("Invalid segmentation level. "
                                     "Valid options: 'word', 'bpe', 'char'.")
        self.shuffle = combiner_train_config.get("shuffle", True)
        self.epochs = combiner_train_config["epochs"]
        self.batch_size = combiner_train_config["batch_size"]
        # Placeholder so that we can use the train_iter in other functions.
        self.train_iter = None
        self.train_iter_state = None
        # per-device batch_size = self.batch_size // self.n_gpu
        self.batch_type = combiner_train_config.get("batch_type", "sentence")
        self.eval_batch_size = combiner_train_config.get("eval_batch_size",
                                                self.batch_size)
        # per-device eval_batch_size = self.eval_batch_size // self.n_gpu
        self.eval_batch_type = combiner_train_config.get("eval_batch_type",
                                                self.batch_type)

        self.batch_multiplier = combiner_train_config.get("batch_multiplier", 1)

        # generation
        self.max_output_length = combiner_train_config.get("max_output_length", None)

        # CPU / GPU
        self.use_cuda = combiner_train_config["use_cuda"] and torch.cuda.is_available()
        self.n_gpu = torch.cuda.device_count() if self.use_cuda else 0
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.model.to(self.device)

        # fp16
        self.fp16 = combiner_train_config.get("fp16", False)
        if self.fp16:
            if 'apex' not in sys.modules:
                raise ImportError("Please install apex from "
                                  "https://www.github.com/nvidia/apex "
                                  "to use fp16 training.") from no_apex
            self.model, self.optimizer = amp.initialize(self.model,
                                                        self.optimizer,
                                                        opt_level='O1')
            # opt level: one of {"O0", "O1", "O2", "O3"}
            # see https://nvidia.github.io/apex/amp.html#opt-levels

        # initialize training statistics
        self.stats = self.TrainStatistics(
            steps=0,
            stop=False,
            total_tokens=0,
            best_ckpt_iter=0,
            best_ckpt_score=np.inf if self.minimize_metric else -np.inf,
            minimize_metric=self.minimize_metric,
            max_steps=combiner_train_config["max_steps"])

        # model parameters
        if "load_model" in combiner_train_config.keys():
            self.init_from_checkpoint(
                combiner_train_config["load_model"],
                reset_best_ckpt=combiner_train_config.get("reset_best_ckpt", False),
                reset_scheduler=combiner_train_config.get("reset_scheduler", False),
                reset_optimizer=combiner_train_config.get("reset_optimizer", False),
                reset_iter_state=combiner_train_config.get("reset_iter_state", False))

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            self.model = _DataParallel(self.model)

    def _save_checkpoint(self, new_best: bool = True) -> None:
        """
        Save the model's current parameters and the training state to a
        checkpoint.

        The training state contains the total number of training steps,
        the total number of training tokens,
        the best checkpoint score and iteration so far,
        and optimizer and scheduler states.

        :param new_best: This boolean signals which symlink we will use for the
          new checkpoint. If it is true, we update best.ckpt, else latest.ckpt.
        """
        combiner_path = os.path.join(self.model_dir,
                                  "{}.ckpt".format(self.stats.steps))
        combiner_state_dict = self.model.combiner.module.state_dict() \
            if isinstance(self.model.combiner, torch.nn.DataParallel) \
            else self.model.combiner.state_dict()
        state = {
            "steps":
            self.stats.steps,
            "total_tokens":
            self.stats.total_tokens,
            "best_ckpt_score":
            self.stats.best_ckpt_score,
            "best_ckpt_iteration":
            self.stats.best_ckpt_iter,
            "model_state":
            combiner_state_dict,
            "optimizer_state":
            self.optimizer.state_dict(),
            "scheduler_state":
            self.scheduler.state_dict() if self.scheduler is not None else None,
            'amp_state':
            amp.state_dict() if self.fp16 else None,
            "train_iter_state":
            self.train_iter.state_dict()
        }
        torch.save(state, combiner_path)
        symlink_target = "{}.ckpt".format(self.stats.steps)
        if new_best:
            if len(self.ckpt_queue) == self.ckpt_queue.maxlen:
                to_delete = self.ckpt_queue.popleft()  # delete oldest ckpt
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    logger.warning(
                        "Wanted to delete old checkpoint %s but "
                        "file does not exist.", to_delete)

            self.ckpt_queue.append(combiner_path)

            best_path = "{}/best.ckpt".format(self.model_dir)
            try:
                # create/modify symbolic link for best checkpoint
                symlink_update(symlink_target, best_path)
            except OSError:
                # overwrite best.ckpt
                torch.save(state, best_path)

        if self.save_latest_checkpoint:
            last_path = "{}/latest.ckpt".format(self.model_dir)
            previous_path = latest_checkpoint_update(symlink_target, last_path)
            # If the last ckpt is in the ckpt_queue, we don't want to delete it.
            can_delete = True
            for ckpt_path in self.ckpt_queue:
                if pathlib.Path(ckpt_path).resolve() == previous_path:
                    can_delete = False
                    break
            if can_delete and previous_path is not None:
                os.remove(previous_path)

    def init_from_checkpoint(self,
                             path: str,
                             reset_best_ckpt: bool = False,
                             reset_scheduler: bool = False,
                             reset_optimizer: bool = False,
                             reset_iter_state: bool = False) -> None:
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also
        scheduler and optimizer states, see `self._save_checkpoint`.

        :param path: path to checkpoint
        :param reset_best_ckpt: reset tracking of the best checkpoint,
                                use for domain adaptation with a new dev
                                set or when using a new metric for fine-tuning.
        :param reset_scheduler: reset the learning rate scheduler, and do not
                                use the one stored in the checkpoint.
        :param reset_optimizer: reset the optimizer, and do not use the one
                                stored in the checkpoint.
        :param reset_iter_state: reset the sampler's internal state and do not
                                use the one stored in the checkpoint.
        """
        logger.info("Loading model from %s", path)
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        else:
            logger.info("Reset optimizer.")

        if not reset_scheduler:
            if model_checkpoint["scheduler_state"] is not None and \
                    self.scheduler is not None:
                self.scheduler.load_state_dict(
                    model_checkpoint["scheduler_state"])
        else:
            logger.info("Reset scheduler.")

        # restore counts
        self.stats.steps = model_checkpoint["steps"]
        self.stats.total_tokens = model_checkpoint["total_tokens"]

        if not reset_best_ckpt:
            self.stats.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.stats.best_ckpt_iter = model_checkpoint["best_ckpt_iteration"]
        else:
            logger.info("Reset tracking of the best checkpoint.")

        if (not reset_iter_state
                and model_checkpoint.get('train_iter_state', None) is not None):
            self.train_iter_state = model_checkpoint["train_iter_state"]

        # move parameters to cuda
        if self.use_cuda:
            self.model.to(self.device)

        # fp16
        if self.fp16 and model_checkpoint.get("amp_state", None) is not None:
            amp.load_state_dict(model_checkpoint['amp_state'])

    # pylint: disable=unnecessary-comprehension
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    def train_and_validate(self, train_data: Dataset, valid_data: Dataset) \
            -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """
        self.train_iter = make_data_iter(train_data,
                                         batch_size=self.batch_size,
                                         batch_type=self.batch_type,
                                         train=True,
                                         shuffle=self.shuffle)

        if self.train_iter_state is not None:
            self.train_iter.load_state_dict(self.train_iter_state)

        #################################################################
        # simplify accumulation logic:
        #################################################################
        # for epoch in range(epochs):
        #     self.model.zero_grad()
        #     epoch_loss = 0.0
        #     batch_loss = 0.0
        #     for i, batch in enumerate(iter(self.train_iter)):
        #
        #         # gradient accumulation:
        #         # loss.backward() inside _train_step()
        #         batch_loss += self._train_step(inputs)
        #
        #         if (i + 1) % self.batch_multiplier == 0:
        #             self.optimizer.step()     # update!
        #             self.model.zero_grad()    # reset gradients
        #             self.steps += 1           # increment counter
        #
        #             epoch_loss += batch_loss  # accumulate batch loss
        #             batch_loss = 0            # reset batch loss
        #
        #     # leftovers are just ignored.
        #################################################################

        logger.info(
            "Train stats:\n"
            "\tdevice: %s\n"
            "\tn_gpu: %d\n"
            "\t16-bits training: %r\n"
            "\tgradient accumulation: %d\n"
            "\tbatch size per device: %d\n"
            "\ttotal batch size (w. parallel & accumulation): %d", self.device,
            self.n_gpu, self.fp16, self.batch_multiplier, self.batch_size //
            self.n_gpu if self.n_gpu > 1 else self.batch_size,
            self.batch_size * self.batch_multiplier)
        
        self.model.eval()

        for epoch_no in range(self.epochs):
            logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.combiner.train()

            # Reset statistics for each epoch.
            start = time.time()
            total_valid_duration = 0
            start_tokens = self.stats.total_tokens
            self.model.combiner.zero_grad()
            epoch_loss = 0
            batch_loss = 0

            for i, batch in enumerate(iter(self.train_iter)):
                # create a Batch object from torchtext batch
                batch = self.batch_class(batch, self.model.pad_index,
                                         use_cuda=self.use_cuda)

                # get batch loss
                batch_loss += self._train_step(batch)

                # update!
                if (i + 1) % self.batch_multiplier == 0:
                    # clip gradients (in-place)
                    if self.clip_grad_fun is not None:
                        if self.fp16:
                            self.clip_grad_fun(
                                params=amp.master_params(self.optimizer))
                        else:
                            self.clip_grad_fun(params=self.model.combiner.parameters())

                    # make gradient step
                    self.optimizer.step()

                    # decay lr
                    if self.scheduler is not None \
                            and self.scheduler_step_at == "step":
                        self.scheduler.step()

                    # reset gradients
                    self.model.combiner.zero_grad()

                    # increment step counter
                    self.stats.steps += 1
                    if self.stats.steps >= self.stats.max_steps:
                        self.stats.stop = True

                    # log learning progress
                    if self.stats.steps % self.logging_freq == 0:
                        self.tb_writer.add_scalar("train/train_batch_loss",
                                                  batch_loss, self.stats.steps)
                        elapsed = time.time() - start - total_valid_duration
                        elapsed_tokens = self.stats.total_tokens - start_tokens
                        logger.info(
                            "Epoch %3d, Step: %8d, Batch Loss: %12.6f, "
                            "Tokens per Sec: %8.0f, Lr: %.6f", epoch_no + 1,
                            self.stats.steps, batch_loss,
                            elapsed_tokens / elapsed,
                            self.optimizer.param_groups[0]["lr"])
                        start = time.time()
                        total_valid_duration = 0
                        start_tokens = self.stats.total_tokens

                    # Only add complete loss of full mini-batch to epoch_loss
                    epoch_loss += batch_loss  # accumulate epoch_loss
                    batch_loss = 0  # rest batch_loss

                    # validate on the entire dev set
                    if self.stats.steps % self.validation_freq == 0:
                        valid_duration = self._validate(valid_data, epoch_no)
                        total_valid_duration += valid_duration

                if self.stats.stop:
                    break
            if self.stats.stop:
                logger.info('Training ended since minimum lr %f was reached.',
                            self.learning_rate_min)
                break

            logger.info('Epoch %3d: total training loss %.2f', epoch_no + 1,
                        epoch_loss)
        else:
            logger.info('Training ended after %3d epochs.', epoch_no + 1)
        logger.info('Best validation result (greedy) at step %8d: %6.2f %s.',
                    self.stats.best_ckpt_iter, self.stats.best_ckpt_score,
                    self.early_stopping_metric)

        self.tb_writer.close()  # close Tensorboard writer

    def _train_step(self, batch: Batch) -> Tensor:
        """
        Train the model on one batch: Compute the loss.

        :param batch: training batch
        :return: loss for batch (sum)
        """
        # reactivate training
        self.model.combiner.train()

        # get loss
        batch_loss, _, _, _ = self.model(return_type="combiner_loss", **vars(batch))

        # sum multi-gpu losses
        if self.n_gpu > 1:
            batch_loss = batch_loss.sum()

        # normalize batch loss
        if self.normalization == "batch":
            normalizer = batch.nseqs
        elif self.normalization == "tokens":
            normalizer = batch.ntokens
        elif self.normalization == "none":
            normalizer = 1
        else:
            raise NotImplementedError("Only normalize by 'batch' or 'tokens' "
                                      "or summation of loss 'none' implemented")

        norm_batch_loss = batch_loss / normalizer

        if self.n_gpu > 1:
            norm_batch_loss = norm_batch_loss / self.n_gpu

        if self.batch_multiplier > 1:
            norm_batch_loss = norm_batch_loss / self.batch_multiplier

        # accumulate gradients
        if self.fp16:
            with amp.scale_loss(norm_batch_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            norm_batch_loss.backward()

        # increment token counter
        self.stats.total_tokens += batch.ntokens

        return norm_batch_loss.item()

    def _validate(self, valid_data, epoch_no):
        valid_start_time = time.time()

        valid_score, valid_loss, valid_ppl, valid_sources, \
        valid_sources_raw, valid_references, valid_hypotheses, \
        valid_hypotheses_raw, valid_attention_scores = \
            validate_on_data(
                batch_size=self.eval_batch_size,
                batch_class=self.batch_class,
                data=valid_data,
                eval_metric=self.eval_metric,
                level=self.level, model=self.model,
                use_cuda=self.use_cuda,
                max_output_length=self.max_output_length,
                compute_loss=True,
                beam_size=1,                # greedy validations
                batch_type=self.eval_batch_type,
                postprocess=True,           # always remove BPE for validation
                bpe_type=self.bpe_type,     # "subword-nmt" or "sentencepiece"
                sacrebleu=self.sacrebleu,   # sacrebleu options
                n_gpu=self.n_gpu
            )

        self.tb_writer.add_scalar("valid/valid_loss", valid_loss,
                                  self.stats.steps)
        self.tb_writer.add_scalar("valid/valid_score", valid_score,
                                  self.stats.steps)
        self.tb_writer.add_scalar("valid/valid_ppl", valid_ppl,
                                  self.stats.steps)

        if self.early_stopping_metric == "loss":
            ckpt_score = valid_loss
        elif self.early_stopping_metric in ["ppl", "perplexity"]:
            ckpt_score = valid_ppl
        else:
            ckpt_score = valid_score

        if self.scheduler is not None \
                and self.scheduler_step_at == "validation":
            self.scheduler.step(ckpt_score)

        new_best = False
        if self.stats.is_best(ckpt_score):
            self.stats.best_ckpt_score = ckpt_score
            self.stats.best_ckpt_iter = self.stats.steps
            logger.info('Hooray! New best validation result [%s]!',
                        self.early_stopping_metric)
            if self.ckpt_queue.maxlen > 0:
                logger.info("Saving new checkpoint.")
                new_best = True
                self._save_checkpoint(new_best)
        elif self.save_latest_checkpoint:
            self._save_checkpoint(new_best)

        # append to validation report
        self._add_report(valid_score=valid_score,
                         valid_loss=valid_loss,
                         valid_ppl=valid_ppl,
                         eval_metric=self.eval_metric,
                         new_best=new_best)

        self._log_examples(sources_raw=[v for v in valid_sources_raw],
                           sources=valid_sources,
                           hypotheses_raw=valid_hypotheses_raw,
                           hypotheses=valid_hypotheses,
                           references=valid_references)

        valid_duration = time.time() - valid_start_time
        logger.info(
            'Validation result (greedy) at epoch %3d, '
            'step %8d: %s: %6.2f, loss: %8.4f, ppl: %8.4f, '
            'duration: %.4fs', epoch_no + 1, self.stats.steps, self.eval_metric,
            valid_score, valid_loss, valid_ppl, valid_duration)

        # store validation set outputs
        self._store_outputs(valid_hypotheses)

        # store attention plots for selected valid sentences
        if valid_attention_scores:
            store_attention_plots(attentions=valid_attention_scores,
                                  targets=valid_hypotheses_raw,
                                  sources=[s for s in valid_data.src],
                                  indices=self.log_valid_sents,
                                  output_prefix="{}/att.{}".format(
                                      self.model_dir, self.stats.steps),
                                  tb_writer=self.tb_writer,
                                  steps=self.stats.steps)

        return valid_duration

    def _add_report(self,
                    valid_score: float,
                    valid_ppl: float,
                    valid_loss: float,
                    eval_metric: str,
                    new_best: bool = False) -> None:
        """
        Append a one-line report to validation logging file.

        :param valid_score: validation evaluation score [eval_metric]
        :param valid_ppl: validation perplexity
        :param valid_loss: validation loss (sum over whole validation set)
        :param eval_metric: evaluation metric, e.g. "bleu"
        :param new_best: whether this is a new best model
        """
        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        if current_lr < self.learning_rate_min:
            self.stats.stop = True

        with open(self.valid_report_file, 'a') as opened_file:
            opened_file.write(
                "Steps: {}\tLoss: {:.5f}\tPPL: {:.5f}\t{}: {:.5f}\t"
                "LR: {:.8f}\t{}\n".format(self.stats.steps, valid_loss,
                                          valid_ppl, eval_metric, valid_score,
                                          current_lr, "*" if new_best else ""))

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        combiner_parameters = filter(lambda p: p.requires_grad,
                                  self.model.combiner.parameters())
        n_params = sum([np.prod(p.size()) for p in combiner_parameters])
        logger.info("Total params: %d", n_params)
        trainable_params = [
            n for (n, p) in self.model.combiner.named_parameters() if p.requires_grad
        ]
        logger.debug("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def _log_examples(self,
                      sources: List[str],
                      hypotheses: List[str],
                      references: List[str],
                      sources_raw: List[List[str]] = None,
                      hypotheses_raw: List[List[str]] = None,
                      references_raw: List[List[str]] = None) -> None:
        """
        Log a the first `self.log_valid_sents` sentences from given examples.

        :param sources: decoded sources (list of strings)
        :param hypotheses: decoded hypotheses (list of strings)
        :param references: decoded references (list of strings)
        :param sources_raw: raw sources (list of list of tokens)
        :param hypotheses_raw: raw hypotheses (list of list of tokens)
        :param references_raw: raw references (list of list of tokens)
        """
        for p in self.log_valid_sents:

            if p >= len(sources):
                continue

            logger.info("Example #%d", p)

            if sources_raw is not None:
                logger.debug("\tRaw source:     %s", sources_raw[p])
            if references_raw is not None:
                logger.debug("\tRaw reference:  %s", references_raw[p])
            if hypotheses_raw is not None:
                logger.debug("\tRaw hypothesis: %s", hypotheses_raw[p])

            logger.info("\tSource:     %s", sources[p])
            logger.info("\tReference:  %s", references[p])
            logger.info("\tHypothesis: %s", hypotheses[p])

    def _store_outputs(self, hypotheses: List[str]) -> None:
        """
        Write current validation outputs to file in `self.model_dir.`

        :param hypotheses: list of strings
        """
        current_valid_output_file = "{}/{}.hyps".format(self.model_dir,
                                                        self.stats.steps)
        with open(current_valid_output_file, 'w') as opened_file:
            for hyp in hypotheses:
                opened_file.write("{}\n".format(hyp))

    class TrainStatistics:
        def __init__(self,
                     steps: int = 0,
                     stop: bool = False,
                     total_tokens: int = 0,
                     best_ckpt_iter: int = 0,
                     best_ckpt_score: float = np.inf,
                     minimize_metric: bool = True,
                     max_steps: int = 200000) -> None:
            # global update step counter
            self.steps = steps
            # stop training if this flag is True
            # by reaching learning rate minimum
            self.stop = stop
            # number of total tokens seen so far
            self.total_tokens = total_tokens
            # store iteration point of best ckpt
            self.best_ckpt_iter = best_ckpt_iter
            # initial values for best scores
            self.best_ckpt_score = best_ckpt_score
            # minimize or maximize score
            self.minimize_metric = minimize_metric
            self.max_steps = max_steps

        def is_best(self, score):
            if self.minimize_metric:
                is_best = score < self.best_ckpt_score
            else:
                is_best = score > self.best_ckpt_score
            return is_best


def combiner_train(cfg_file: str, ckpt: str, combiner_cfg: dict) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    """
    cfg = load_config(cfg_file)

    # make logger
    model_dir = make_model_dir(cfg["combiner_training"]["model_dir"],
                               overwrite=cfg["combiner_training"].get(
                                   "overwrite", False))
    _ = make_logger(model_dir, mode="train")  # version string returned
    # TODO: save version number in model checkpoints

    check_combiner_cfg(combiner_cfg)
    cfg["combiner"] = combiner_cfg

    # set the random seed
    set_seed(seed=cfg["combiner_training"].get("random_seed", 42))

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(
        data_cfg=cfg["data"])

    # load model state from disk
    use_cuda = cfg["combiner_training"]["use_cuda"]
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    for p in model.parameters():
        p.requires_grad = False

    combiner = build_combiner(cfg)
    # load combiner from checkpoint for dynamic combiners

    model.combiner = combiner

    # for training management, e.g. early stopping and model selection
    trainer = CombinerTrainManager(model=model, config=cfg)

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, model_dir + "/config.yaml")

    # log all entries of config
    log_cfg(cfg)

    log_data_info(train_data=train_data,
                  valid_data=dev_data,
                  test_data=test_data,
                  src_vocab=src_vocab,
                  trg_vocab=trg_vocab)

    logger.info(str(model))
    logger.info(str(combiner))

    # store the vocabs
    src_vocab_file = "{}/src_vocab.txt".format(cfg["combiner_training"]["model_dir"])
    src_vocab.to_file(src_vocab_file)
    trg_vocab_file = "{}/trg_vocab.txt".format(cfg["combiner_training"]["model_dir"])
    trg_vocab.to_file(trg_vocab_file)

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)