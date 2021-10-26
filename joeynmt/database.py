# -*- coding: utf-8 -*-
# create@ 2021-01-26 18:02

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Tuple
import numpy as np
from joeynmt.faiss_index import FaissIndex

class Database(object):
    """
    Initialize this class with index path, which is built offline,
    and token path which mapping retrieval indices to token id.
    """

    def __init__(self, index_path: str, token_path: str, nprobe: int = 16) -> None:
        super(Database, self).__init__()
        self.index = FaissIndex(load_index_path=index_path, use_gpu=True)
        self.index.set_probe(nprobe)
        self.token_map = self.load_token_mapping(token_path)

    @staticmethod
    def load_token_mapping(token_path: str) -> np.ndarray:
        """
        This function is used to load token mapping from a text file.

        :param token_path: the path of token_map text file.

        :return token_map: np.ndarray shape (index_size,) 
            token_map[i] is the index of i-th embedding in the vocabulary.
        """
        with open(token_path) as f:
            token_map = [int(token_id) for token_id in f.readlines()]
        token_map = np.asarray(token_map).astype(np.int32)
        return token_map

    def search(self, embeddings: np.ndarray, top_k: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function is use to search nearest top_k embeddings from the Faiss index.

        :param embeddings: np.ndarray (batch_size, d)
        :param top_k: int 

        :return distances: np.ndarray (batch_size, top_k)
        :return token_indices: np.ndarray (batch_size, top_k)
        """

        # D, I has shape of batch * top_k
        distances, indices = self.index.search(embeddings, top_k)
        token_indices = self.token_map[indices]
        return distances, token_indices

class EnhancedDatabase(Database):

    def __init__(self, index_path: str, token_path: str, embedding_path: str, in_memory: bool = True, nprobe: int = 16) -> None:
        super(EnhancedDatabase, self).__init__(index_path, token_path, nprobe)
        if in_memory:
            self.embeddings = np.load(embedding_path)
        else:
            self.embeddings = np.load(embedding_path, mmap_mode="r")
    
    def enhanced_search(self, embeddings: np.ndarray, top_k: int = 16, retrieval_dropout: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function is use to search nearest top_k embeddings from the Faiss index.

        :param embeddings: np.ndarray (batch_size, d)
        :param top_k: int 
        :param mask_num: int

        :return distances: np.ndarray (batch_size, top_k)
        :return token_indices: np.ndarray (batch_size, top_k)
        :return hidden: np.ndarray (batch_size, top_k, d)
        """

        # D, I has shape of batch * top_k
        if retrieval_dropout:
            distances, indices = self.index.search(embeddings, top_k + 1)
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        else:
            distances, indices = self.index.search(embeddings, top_k)
        
        token_indices = self.token_map[indices]
        batch_size = indices.shape[0]
        indices = indices.reshape(-1)
        hidden = self.embeddings[indices]
        d = hidden.shape[-1]
        hidden = hidden.reshape(batch_size, top_k, d)
        return distances, token_indices, hidden