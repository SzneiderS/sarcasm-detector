from torch.utils.data.sampler import Sampler
from random import shuffle as random_shuffle
from itertools import cycle


class EqualClassSampler(Sampler):
    def __init__(self, dataset, shuffle=False):
        super(EqualClassSampler, self).__init__(None)

        self.dataset = dataset
        self.shuffle = shuffle

        self.class_indices = {}
        self.class_num = 0
        self.classes = set()
        self._calculate_classes()

    def _calculate_classes(self):
        for n, (_, t) in enumerate(self.dataset):
            t = str(t)
            if t not in self.class_indices:
                self.classes.add(t)
                self.class_indices[t] = []
            self.class_indices[t].append(n)
        self.class_num = len(self.class_indices)
        for c in self.classes:
            random_shuffle(self.class_indices[c])

    def __iter__(self):
        current_class = cycle(range(self.class_num))
        class_iters = []
        for c in self.classes:
            class_iters.append(cycle(iter(self.class_indices[c])))
        for _ in range(len(self.dataset)):
            yield next(class_iters[next(current_class)])
        if self.shuffle:
            for c in self.classes:
                random_shuffle(self.class_indices[c])
