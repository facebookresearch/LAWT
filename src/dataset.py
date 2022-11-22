# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
import os
import io
import sys
import numpy as np


import torch
from torch.utils.data.dataset import Dataset

logger = getLogger()


class EnvDataset(Dataset):
    def __init__(self, env, task, train, params, path, size=None, type=None):
        super(EnvDataset).__init__()
        self.env = env
        self.train = train
        self.task = task
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.global_rank = params.global_rank
        self.count = 0
        self.type = type
        assert size is None or not self.train
        assert not params.batch_load or params.reload_size > 0

        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size

        self.batch_load = params.batch_load
        self.reload_size = params.reload_size
        self.local_rank = params.local_rank
        self.n_gpu_per_node = params.n_gpu_per_node

        self.basepos = 0
        self.nextpos = 0
        self.seekpos = 0

        # generation, or reloading from file
        if path is not None:
            assert os.path.isfile(path)
            if params.batch_load and self.train:
                self.load_chunk()
            else:
                logger.info(f"Loading data from {path} ...")
                with io.open(path, mode="r", encoding="utf-8") as f:
                    # either reload the entire file, or the first N lines
                    # (for the training set)
                    if not train:
                        lines = [line.rstrip().split("|") for line in f]
                    else:
                        lines = []
                        for i, line in enumerate(f):
                            if i == params.reload_size:
                                break
                            if i % params.n_gpu_per_node == params.local_rank:
                                lines.append(line.rstrip().split("|"))
                self.data = [xy.split("\t") for _, xy in lines]
                self.data = [xy for xy in self.data if len(xy) == 2]
                logger.info(f"Loaded {len(self.data)} equations from the disk.")

        # dataset size: infinite iterator for train, finite for valid / test
        # (default of 10000 if no file provided)
        if self.train:
            self.size = 1 << 60
        elif size is None:
            self.size = 10000 if path is None else len(self.data)
        else:
            assert size > 0
            self.size = size

    def load_chunk(self):
        self.basepos = self.nextpos
        logger.info(
            f"Loading data from {self.path} ... seekpos {self.seekpos}, "
            f"basepos {self.basepos}"
        )
        endfile = False
        with io.open(self.path, mode="r", encoding="utf-8") as f:
            f.seek(self.seekpos, 0)
            lines = []
            for i in range(self.reload_size):
                line = f.readline()
                if not line:
                    endfile = True
                    break
                if i % self.n_gpu_per_node == self.local_rank:
                    lines.append(line.rstrip().split("|"))
            self.seekpos = 0 if endfile else f.tell()

        self.data = [xy.split("\t") for _, xy in lines]
        self.data = [xy for xy in self.data if len(xy) == 2]
        self.nextpos = self.basepos + len(self.data)
        logger.info(
            f"Loaded {len(self.data)} equations from the disk. seekpos {self.seekpos}, "
            f"nextpos {self.nextpos}"
        )
        if len(self.data) == 0:
            self.load_chunk()

    def batch_sequences(self, sequences, pad_index, eos_index):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(
            pad_index
        )
        assert lengths.min().item() > 2

        sent[0] = eos_index
        for i, s in enumerate(sequences):
            sent[1 : lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = eos_index

        return sent, lengths

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        x, y = zip(*elements)
        nb_eqs = [self.env.code_class(xi, yi) for xi, yi in zip(x, y)]
        x = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in x]
        y = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in y]
        x, x_len = self.batch_sequences(x, self.env.pad_index, self.env.eos_index)
        y, y_len = self.batch_sequences(y, self.env.pad_index, self.env.eos_index)
        return (x, x_len), (y, y_len), torch.LongTensor(nb_eqs)

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if hasattr(self.env, "rng"):
            return
        if self.train:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            self.env.rng = np.random.RandomState(
                [worker_id, self.global_rank, self.env_base_seed]
            )
            logger.info(
                f"Initialized random generator for worker {worker_id}, with seed "
                f"{[worker_id, self.global_rank, self.env_base_seed]} "
                f"(base seed={self.env_base_seed})."
            )
        else:
            self.env.rng = np.random.RandomState(None) # if self.type == "valid" else 0)

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        if self.path is None:
            return self.generate_sample()
        else:
            return self.read_sample(index)

    def read_sample(self, index):
        """
        Read a sample.
        """
        idx = index
        if self.train:
            if self.batch_load:
                if index >= self.nextpos:
                    self.load_chunk()
                idx = index - self.basepos
            else:
                index = self.env.rng.randint(len(self.data))
                idx = index
        x, y = self.data[idx]
        x = x.split()
        y = y.split()
        assert len(x) >= 1 and len(y) >= 1
        return x, y

    def generate_sample(self):
        """
        Generate a sample.
        """
        while True:
            try:
                xy = self.env.gen_expr(self.type, self.task, self.train)
                if xy is None:
                    continue
                x, y = xy
                break
            except Exception as e:
                logger.error(
                    'An unknown exception of type {0} occurred for worker {4} in line {1} for expression "{2}". Arguments:{3!r}.'.format(
                        type(e).__name__,
                        sys.exc_info()[-1].tb_lineno,
                        "F",
                        e.args,
                        self.get_worker_id(),
                    )
                )
                continue
        self.count += 1

        return x, y
