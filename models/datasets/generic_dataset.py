from torch.utils.data.dataset import Dataset
from typing import Tuple, Any
from random import shuffle


class GenericDataset(Dataset):
    def __init__(self, transforms=None):
        """
        Args:
            transforms (:ref:`torchvision.transforms` or None): transforms applied to input
        """
        self.examples = []
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, item: int) -> Tuple[Any, Any]:
        inp, target = self.examples[item]
        inp, target = self.transform((inp, target))
        return inp, target

    def transform(self, example):
        inp, target = example
        if self.transforms is not None:
            if 'both' in self.transforms:
                inp = self.transforms['both'](inp)
                target = self.transforms['both'](target)
            if 'input' in self.transforms:
                inp = self.transforms['input'](inp)
            if 'target' in self.transforms:
                target = self.transforms['target'](target)
        return inp, target

    @classmethod
    def _standard_creator(cls, examples, transforms=None, train_split: float = 0.8, shuffle_dataset: bool = False,
                          limit: int or None = None):
        train = cls(transforms)
        test = cls(transforms)

        if shuffle_dataset:
            shuffle(examples)

        count = len(examples)

        if limit is not None:
            count = min(count, limit)

        split = int(count * train_split)

        train.examples = examples[0:split]
        test.examples = examples[split:count]

        return train, test
