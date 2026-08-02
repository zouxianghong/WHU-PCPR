# Author: Jacek Komorowski
# Warsaw University of Technology

import random

from torchpack.utils.config import configs 

from torch.utils.data import Sampler

from datasets.oxford import OxfordDataset
from misc.utils import ListDict


class BatchSampler(Sampler):
    # Sampler returning list of indices to form a mini-batch
    # Samples elements in groups consisting of k=2 similar elements (positives)
    # Batch has the following structure: item1_1, ..., item1_k, item2_1, ... item2_k, itemn_1, ..., itemn_k
    def __init__(self, dataset: OxfordDataset, batch_size: int, batch_size_limit: int = None,
                 batch_expansion_rate: float = None, max_batches: int = None):
        if batch_expansion_rate is not None:
            assert batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
            assert batch_size <= batch_size_limit, 'batch_size_limit must be greater or equal to batch_size'

        self.batch_size = batch_size
        self.batch_size_limit = batch_size_limit
        self.batch_expansion_rate = batch_expansion_rate
        self.max_batches = max_batches
        self.dataset = dataset
        self.k = 2  # Number of positive examples per group must be 2
        if self.batch_size < 2 * self.k:
            self.batch_size = 2 * self.k
            print('WARNING: Batch too small. Batch size increased to {}.'.format(self.batch_size))

        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)
        self.elems_ndx = list(self.dataset.queries)    # List of point cloud indexes
        self.database_ndx = list(range(len(self.elems_ndx)-configs.train.memory.num_pairs*2))
        self.memory_ndx = list(range(len(self.database_ndx), len(self.database_ndx)+configs.train.memory.num_pairs*2))  # list of point cloud indexes

    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches()
        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def expand_batch(self):
        if self.batch_expansion_rate is None:
            print('WARNING: batch_expansion_rate is None')
            return

        if self.batch_size >= self.batch_size_limit:
            return

        old_batch_size = self.batch_size
        self.batch_size = int(self.batch_size * self.batch_expansion_rate)
        self.batch_size = min(self.batch_size, self.batch_size_limit)
        print('=> Batch size increased from: {} to {}'.format(old_batch_size, self.batch_size))

    def generate_batches(self):
        # Generate training/evaluation batches.
        # batch_idx holds indexes of elements in each batch as a list of lists
        self.batch_idx = []

        unused_elements_ndx = ListDict(self.elems_ndx)
        current_batch = []

        assert self.k == 2, 'sampler can sample only k=2 elements from the same class'

        while True:
            if len(current_batch) >= self.batch_size or len(unused_elements_ndx) == 0:
                # Flush out batch, when it has a desired size, or a smaller batch, when there's no more
                # elements to process
                if len(current_batch) >= 2*self.k:
                    # Ensure there're at least two groups of similar elements, otherwise, it would not be possible
                    # to find negative examples in the batch
                    assert len(current_batch) % self.k == 0, 'Incorrect bach size: {}'.format(len(current_batch))
                    
                    # select memory batches in each iteration
                    if len(self.memory_ndx) > len(current_batch):  # No Memory Samples when training on the first scene!
                        memory_batch = []
                        while len(memory_batch) < len(current_batch):
                            selected_m = random.choice(self.memory_ndx)
                            if selected_m in memory_batch:
                                continue
                            positives = self.dataset.get_positives(selected_m)
                            if len(positives) == 0:
                                continue
                            second_m_positive = random.choice(list(positives))
                            memory_batch += [selected_m, second_m_positive]
                        current_batch += memory_batch

                    self.batch_idx.append(current_batch)
                    current_batch = []
                    if (self.max_batches is not None) and (len(self.batch_idx) >= self.max_batches):
                        break
                if len(unused_elements_ndx) == 0:
                    break

            # Add k=2 similar elements to the batch
            selected_element = unused_elements_ndx.choose_random()
            unused_elements_ndx.remove(selected_element)
            positives = self.dataset.get_positives(selected_element)
            if len(positives) == 0:
                # Broken dataset element without any positives
                continue

            unused_positives = [e for e in positives if e in unused_elements_ndx]
            # If there're unused elements similar to selected_element, sample from them
            # otherwise sample from all similar elements
            if len(unused_positives) > 0:
                second_positive = random.choice(unused_positives)
                unused_elements_ndx.remove(second_positive)
            else:
                second_positive = random.choice(list(positives))

            current_batch += [selected_element, second_positive]

        for batch in self.batch_idx:
            assert len(batch) % self.k == 0, 'Incorrect bach size: {}'.format(len(batch))
