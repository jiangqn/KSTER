import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import math
from typing import Tuple

from joeynmt.database import Database, EnhancedDatabase
from joeynmt.kernel import Kernel, GaussianKernel, LaplacianKernel

class Combiner(nn.Module):

    def __init__(self) -> None:
        super(Combiner, self).__init__()
    
    def forward(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        :param hidden: hidden states FloatTensor (batch_size, seq_len, hidden_size)
        :param logits: hidden states FloatTensor (batch_size, seq_len, hidden_size)
        :return log_probs: FloatTensor (batch_size, seq_len, vocab_size)
        """
        raise NotImplementedError("The forward method is not implemented in the Combiner class.")

    def detailed_forward(self, hidden: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param hidden: hidden states FloatTensor (batch_size, seq_len, hidden_size)
        :param logits: hidden states FloatTensor (batch_size, seq_len, hidden_size)
        :param trg: true targets for force decoding.
        
        :return mixed_distribution: FloatTensor (batch_size, seq_len, vocab_size)
        :return model_based_distribution: FloatTensor (batch_size, seq_len, vocab_size)
        :return example_based_distribution: FloatTensor (batch_size, seq_len, vocab_size)
        :return mixing_weight: FloatTensor (batch_size, seq_len)
        :return bandwidth: FloatTensor (batch_size, seq_len)
        """
        raise NotImplementedError("The forward method is not implemented in the Combiner class.")

class NoCombiner(Combiner):

    def __init__(self) -> None:
        super(NoCombiner, self).__init__()
    
    def forward(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

class StaticCombiner(Combiner):

    def __init__(self, database: Database, top_k: int, mixing_weight: float, kernel: Kernel, bandwidth: float) -> None:
        super(StaticCombiner, self).__init__()
        self.database = database
        self.top_k = top_k
        self.mixing_weight = mixing_weight
        self.kernel = kernel
        self.bandwidth = bandwidth
    
    def forward(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        
        batch_size, seq_len, hidden_size = hidden.size()
        vocab_size = logits.size(-1)
        hidden = hidden.view(batch_size * seq_len, hidden_size)
        logits = logits.view(batch_size * seq_len, vocab_size)

        model_based_distribution = F.softmax(logits, dim=-1)
        vocab_size = model_based_distribution.size(-1)
        distances, token_indices = self.database.search(hidden.cpu().numpy(), top_k=self.top_k)
        distances = torch.FloatTensor(distances).to(hidden.device)
        token_indices = torch.LongTensor(token_indices).to(hidden.device)
        example_based_distribution, _ = self.kernel.compute_example_based_distribution(distances, self.bandwidth, token_indices, vocab_size)

        mixed_distribution = (1 - self.mixing_weight) * model_based_distribution + self.mixing_weight * example_based_distribution
        log_probs = torch.log(mixed_distribution)
        log_probs = log_probs.view(batch_size, seq_len, vocab_size).contiguous()
        return log_probs

class DynamicCombiner(Combiner):

    def __init__(self, database: EnhancedDatabase, top_k: int, kernel: Kernel) -> None:
        super(DynamicCombiner, self).__init__()
        self.database = database
        self.top_k = top_k
        self.kernel = kernel
        dimension = database.index.index.d
        self.bandwidth_estimator = nn.Linear(2 * dimension, 1)
        if isinstance(kernel, GaussianKernel):
            self.bandwidth_estimator.bias.data[0] = math.log(100)
        else:
            self.bandwidth_estimator.bias.data[0] = math.log(10)
        self.mixing_weight_estimator = nn.Sequential(
            nn.Linear(2 * dimension, dimension),
            nn.ReLU(),
            nn.Linear(dimension, 1)
        )

    def compute_bandwidth(self, hidden: torch.Tensor, searched_hidden: torch.Tensor) -> torch.Tensor:
        """
        :param hidden: torch.FloatTensor (batch_size * seq_len, hidden_size)
        :param searched_hidden: torch.FloatTensor (batch_size * seq_len, top_k, hidden_size)
        
        :return bandwidth: torch.FloatTensor (batch_size * seq_len,)
        """
        mean_hidden = searched_hidden.mean(dim=1)
        bandwidth = torch.exp(self.bandwidth_estimator(torch.cat([hidden, mean_hidden], dim=-1)))
        return bandwidth

    def compute_mixing_weight(self, hidden: torch.Tensor, searched_hidden: torch.Tensor, sparse_probs: torch.Tensor) -> torch.Tensor:
        """
        :param hidden: torch.FloatTensor (batch_size * seq_len, hidden_size)
        :param searched_hidden: torch.FloatTensor (batch_size * seq_len, top_k, hidden_size)
        :param sparse_probs: torch.FloatTensor (batch_size * seq_len, top_k)
        
        :return mixing_weight: torch.FloatTensor (batch_size * seq_len,)
        """
        merged_hidden = searched_hidden.transpose(1, 2).matmul(sparse_probs.unsqueeze(-1)).squeeze(-1)
        mixing_weight = torch.sigmoid(self.mixing_weight_estimator(torch.cat([hidden, merged_hidden], dim=-1)))
        return mixing_weight

    def forward(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        
        # reshape hidden and logits for database retrieval
        batch_size, seq_len, hidden_size = hidden.size()
        vocab_size = logits.size(-1)
        hidden = hidden.view(batch_size * seq_len, hidden_size)
        logits = logits.view(batch_size * seq_len, vocab_size)

        # retrieve examples from database
        if self.training:
            distances, token_indices, searched_hidden = self.database.enhanced_search(hidden.cpu().numpy(), 
                top_k=self.top_k, retrieval_dropout=True)
        else:
            distances, token_indices, searched_hidden = self.database.enhanced_search(hidden.cpu().numpy(), 
                top_k=self.top_k, retrieval_dropout=False)
        
        distances = torch.FloatTensor(distances).to(hidden.device)
        token_indices = torch.LongTensor(token_indices).to(hidden.device)
        searched_hidden = torch.FloatTensor(searched_hidden).to(hidden.device)

        # compute dynamic database bandwidth
        bandwidth = self.compute_bandwidth(hidden, searched_hidden)

        model_based_distribution = F.softmax(logits, dim=-1)
        vocab_size = model_based_distribution.size(-1)
        example_based_distribution, sparse_example_based_distribution = self.kernel.compute_example_based_distribution(distances, 
            bandwidth, token_indices, vocab_size)
        
        mixing_weight = self.compute_mixing_weight(hidden, searched_hidden, sparse_example_based_distribution)

        # compute prediction distribution by interpolating between model distribution and database distribution
        mixed_distribution = (1 - mixing_weight) * model_based_distribution + mixing_weight * example_based_distribution
        log_probs = torch.log(mixed_distribution)

        log_probs = log_probs.view(batch_size, seq_len, vocab_size).contiguous()
        return log_probs
    
    def detailed_forward(self, hidden: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param hidden: pre-softmax hidden states FloatTensor (batch_size, seq_len, hidden_size)
        :param logits: pre-softmax hidden states FloatTensor (batch_size, seq_len, hidden_size)
        :param trg: true targets for force decoding.
        
        :return mixed_distribution: FloatTensor (batch_size, seq_len, vocab_size)
        :return model_based_distribution: FloatTensor (batch_size, seq_len, vocab_size)
        :return example_based_distribution: FloatTensor (batch_size, seq_len, vocab_size)
        :return mixing_weight: FloatTensor (batch_size, seq_len)
        :return bandwidth: FloatTensor (batch_size, seq_len)
        """
        # reshape hidden and logits for knn retrieval
        batch_size, seq_len, hidden_size = hidden.size()
        vocab_size = logits.size(-1)
        hidden = hidden.view(batch_size * seq_len, hidden_size)
        logits = logits.view(batch_size * seq_len, vocab_size)

        # retrieve examples from database
        if self.training:
            distances, token_indices, searched_hidden = self.database.enhanced_search(hidden.cpu().numpy(), 
                top_k=self.top_k, retrieval_dropout=True)
        else:
            distances, token_indices, searched_hidden = self.database.enhanced_search(hidden.cpu().numpy(), 
                top_k=self.top_k, retrieval_dropout=False)
        
        distances = torch.FloatTensor(distances).to(hidden.device)
        token_indices = torch.LongTensor(token_indices).to(hidden.device)
        searched_hidden = torch.FloatTensor(searched_hidden).to(hidden.device)

        # compute dynamic database bandwidth
        bandwidth = self.compute_bandwidth(hidden, searched_hidden)

        model_based_distribution = F.softmax(logits, dim=-1)
        vocab_size = model_based_distribution.size(-1)
        example_based_distribution, sparse_example_based_distribution = self.kernel.compute_example_based_distribution(distances, 
            bandwidth, token_indices, vocab_size)
        
        mixing_weight = self.compute_mixing_weight(hidden, searched_hidden, sparse_example_based_distribution)

        # compute prediction distribution by interpolating between model distribution and database distribution
        mixed_distribution = (1 - mixing_weight) * model_based_distribution + mixing_weight * example_based_distribution

        mixed_distribution = mixed_distribution.view(batch_size, seq_len, vocab_size).contiguous()
        model_based_distribution = model_based_distribution.view(batch_size, seq_len, vocab_size).contiguous()
        example_based_distribution = example_based_distribution.view(batch_size, seq_len, vocab_size).contiguous()
        mixing_weight = mixing_weight.squeeze(-1).view(batch_size, seq_len).contiguous()
        bandwidth = bandwidth.squeeze(-1).view(batch_size, seq_len).contiguous()

        return mixed_distribution, model_based_distribution, example_based_distribution, mixing_weight, bandwidth

def build_combiner(cfg: dict) -> Combiner:

    combiner_cfg = cfg["combiner"]
    combiner_type = combiner_cfg["type"]

    if combiner_type == "no_combiner":

        combiner = NoCombiner()

    elif combiner_type == "static_combiner":

        database = Database(
            index_path=combiner_cfg["index_path"], 
            token_path=combiner_cfg["token_map_path"]
        )

        combiner = StaticCombiner(
            database=database,
            top_k=combiner_cfg["top_k"],
            mixing_weight=combiner_cfg["mixing_weight"],
            bandwidth=combiner_cfg["bandwidth"],
            kernel=GaussianKernel() if combiner_cfg["kernel"] == "gaussian" else LaplacianKernel()
        )

    elif "dynamic" in combiner_type:

        database = EnhancedDatabase(
            index_path=combiner_cfg["index_path"],
            token_path=combiner_cfg["token_map_path"],
            embedding_path=combiner_cfg["embedding_path"],
            in_memory=combiner_cfg["in_memory"]
        )

        combiner = DynamicCombiner(
            database=database,
            top_k=combiner_cfg["top_k"],
            kernel=GaussianKernel() if combiner_cfg["kernel"] == "gaussian" else LaplacianKernel()
        )

    else:

        raise ValueError("The %s is not supported currently." % combiner_type)
    
    return combiner