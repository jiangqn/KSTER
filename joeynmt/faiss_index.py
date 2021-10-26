# -*- coding: utf-8 -*-
# create@ 2021-02-04 13:50

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import faiss
import numpy as np
from typing import Tuple
import re

class FaissIndex(object):

    def __init__(self, factory_template: str = "IVF256,PQ32", load_index_path: str = None, use_gpu: bool = True, index_type: str = "L2") -> None:
        super(FaissIndex, self).__init__()
        self.factory_template = factory_template
        self.gpu_num = faiss.get_num_gpus()
        self.use_gpu = use_gpu and (self.gpu_num > 0)
        self.index_type = index_type
        self._is_trained = False
        if load_index_path != None:
            self.load(index_path=load_index_path)
    
    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _get_clustering_parameters(self, total_samples: int) -> Tuple[int, int]:
        if 0 < total_samples <= 10 ** 6:
            centroids = int(8 * total_samples ** 0.5)
            training_samples = total_samples
        elif 10 ** 6 < total_samples <= 10 ** 7:
            centroids = 65536
            training_samples = min(total_samples, 64 * centroids)
        else:
            centroids = 262144
            training_samples = min(total_samples, 64 * centroids)
        return centroids, training_samples

    def _initialize_index(self, dimension: int, centroids: int) -> faiss.Index:
        template = re.compile(r"IVF\d*").sub(f"IVF{centroids}", self.factory_template)
        if self.index_type == "L2":
            index = faiss.index_factory(dimension, template, faiss.METRIC_L2)
        else:   # self.index_type == "IP"
            index = faiss.index_factory(dimension, template, faiss.METRIC_INNER_PRODUCT)
        if self.use_gpu:
            index = faiss.index_cpu_to_all_gpus(index)
            # index_ivf = faiss.extract_index_ivf(index)
            # if self.index_type == "L2":
            #     clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(dimension))
            # else: # self.index_type == "IP"
            #     clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(dimension))
            # index_ivf.clustering_index = clustering_index
        return index

    def train(self, embeddings_path: str) -> None:
        embeddings = np.load(embeddings_path, mmap_mode="r")
        total_samples, dimension = embeddings.shape
        del embeddings
        centroids, training_samples = self._get_clustering_parameters(total_samples)
        self.index = self._initialize_index(dimension, centroids)
        training_embeddings = self._get_training_embeddings(embeddings_path, training_samples).astype(np.float32)
        self.index.train(training_embeddings)
        self._is_trained = True
    
    def _get_training_embeddings(self, embeddings_path: str, training_samples: int) -> np.ndarray:
        embeddings = np.load(embeddings_path, mmap_mode="r")
        total_samples = embeddings.shape[0]
        sample_indices = np.random.choice(total_samples, training_samples, replace=False)
        sample_indices.sort()
        training_embeddings = embeddings[sample_indices]
        return training_embeddings

    def add(self, embeddings_path: str, batch_size: int = 10000) -> None:
        assert self.is_trained
        embeddings = np.load(embeddings_path)
        total_samples = embeddings.shape[0]
        for i in range(0, total_samples, batch_size):
            start = i
            end = min(total_samples, i + batch_size)
            batch_embeddings = embeddings[start: end].astype(np.float32)
            self.index.add(batch_embeddings)
        del embeddings

    def load(self, index_path: str) -> faiss.Index:
        self.index = faiss.read_index(index_path)
        if self.use_gpu:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        self._is_trained = True

    def export(self, index_path: str) -> None:
        assert self.is_trained
        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(self.index)
        else:
            index = self.index
        faiss.write_index(index, index_path)

    def search(self, embeddings: np.ndarray, top_k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        assert self.is_trained
        distances, indices = self.index.search(embeddings, k=top_k)
        return distances, indices

    def set_probe(self, nprobe):
        self.index.nprobe = nprobe

    @property
    def total(self):
        """
        inspect index volume
        :return:
        """
        return self.index.ntotal