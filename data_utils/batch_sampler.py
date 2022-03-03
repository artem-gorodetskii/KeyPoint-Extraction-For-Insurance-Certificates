# -*- coding: utf-8 -*-

import math
import torch

class SortedSampler(torch.utils.data.Sampler):
    """
    Adapted from https://github.com/PetrochukM/PyTorch-NLP
    Copyright (c) James Bradbury and Soumith Chintala 2016,
    All rights reserved.
    """

    def __init__(self, data, sort_key):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1], reverse=True)
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)

    
class BucketBatchSampler(torch.utils.data.BatchSampler):
    """
    Adapted from https://github.com/PetrochukM/PyTorch-NLP
    Copyright (c) James Bradbury and Soumith Chintala 2016,
    All rights reserved.
    """

    def __init__(
        self,
        sampler,
        batch_size,
        drop_last,
        sort_key,
        bucket_size_multiplier,
    ):
        super().__init__(sampler, batch_size, drop_last)
        self.sort_key = sort_key
        self.bucket_sampler = torch.utils.data.BatchSampler(
            sampler, min(batch_size * bucket_size_multiplier, len(sampler)), False
        )

    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler(bucket, self.sort_key)
            for batch in torch.utils.data.SubsetRandomSampler(
                list(
                    torch.utils.data.BatchSampler(
                        sorted_sampler, self.batch_size, self.drop_last
                    )
                )
            ):
                yield [bucket[i] for i in batch]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)