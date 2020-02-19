from models import BaseModule
import models

from typing import Tuple, Callable, Dict

import torch
import torch.optim as optimizers
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class BaseTrainer:
    def __init__(self, net: BaseModule, train_set: Dataset, test_set: Dataset or None = None, **kwargs):
        if net is not None:
            self.net = net

        self.print_messages: bool = kwargs.get('print_messages', True)

        self.use_cuda: bool = kwargs.get('use_cuda', True) and torch.cuda.is_available()

        self.save_every_nth_epoch: int = kwargs.get('save_every_nth_epoch', 15)
        self.save_name: str = kwargs.get('save_name', None)
        self.load_name: str = kwargs.get('load_name', None)
        self.force_full_load: bool = kwargs.get('force_full_load', False)

        self.train_set = train_set
        self.test_set = test_set

        self.batch_size: int = kwargs.get('batch_size', 1)
        train_sampler = kwargs.get('train_sampler', None)
        batch_train_sampler = kwargs.get('batch_train_sampler', None)

        if train_sampler is not None:
            self.train_loader = DataLoader(train_set, batch_size=self.batch_size, sampler=train_sampler, num_workers=4,
                                           drop_last=True)
        elif batch_train_sampler is not None:
            self.train_loader = DataLoader(train_set, batch_sampler=batch_train_sampler, num_workers=4)
        else:
            self.train_loader = DataLoader(train_set, batch_size=self.batch_size, drop_last=True, shuffle=True,
                                           num_workers=4)
        if test_set is not None:
            test_sampler = kwargs.get('test_sampler', None)
            batch_test_sampler = kwargs.get('batch_test_sampler', None)
            if test_sampler is not None:
                self.test_loader = DataLoader(test_set, batch_size=self.batch_size, sampler=test_sampler, num_workers=4,
                                              drop_last=False)
            elif batch_test_sampler is not None:
                self.test_loader = DataLoader(test_set, batch_sampler=batch_test_sampler, num_workers=4)
            else:
                self.test_loader = DataLoader(test_set, batch_size=self.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4)

        self.accuracy_threshold: float = kwargs.get('accuracy_threshold', 0.01)

        self.accuracy_stop: float = kwargs.get('accuracy_stop', 0.97)

        self.optimizer = None
        self.scheduler = None

        self.lr: float = kwargs.get('lr', 1e-3)

        self.overfit_batch_loss: float or None = kwargs.get('overfit_batch_loss', None)

        self.epoch: int = 1

        # callbacks
        self.after_epoch_func: Callable[[Dict[str, Dict[str, float]]], None] or None = None
        self.after_train_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None] or None = None
        self.after_test_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None] or None = None
        self.is_example_correct_func: Callable[[torch.Tensor, torch.Tensor], bool] or None = None

    def load_network(self) -> None:
        if self.load_name is not None:
            if self.net is None or self.force_full_load:
                self.net = BaseModule.load(self.load_name)
                if self.print_messages:
                    print("Network loaded")
            else:
                if self.net.load_state(self.load_name) and self.print_messages:
                    print("Network loaded")

    def save_network(self) -> None:
        if self.save_name is not None:
            self.net.save(self.save_name)
            if self.print_messages:
                print("Network saved")

    def create_optimizer(self):
        return optimizers.Adam(self.net.parameters(), lr=self.lr)

    @staticmethod
    def create_scheduler(optimizer):
        return optimizers.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, eps=0, patience=5)

    def _overfit_first_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        if self.print_messages:
            print("Overfitting first batch...")
        while True:
            loss, _ = self.single_train_iteration(inputs, targets)
            if self.print_messages:
                print(loss)
            if loss < self.overfit_batch_loss:
                break

    def run(self, max_epochs=None) -> None:
        self.epoch = 1
        self.load_network()

        if self.use_cuda:
            self.net = self.net.cuda()

        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler(self.optimizer)

        if len(self.train_set) > 0:
            if self.overfit_batch_loss is not None:
                for (inputs, targets) in self.train_loader:
                    self.net.train(True)
                    self._overfit_first_batch(inputs, targets)
                    self.net.train(False)
                    break

            while True:
                train_loss = self.train()

                test_accuracy = None
                test_correct = None
                test_loss = None
                if self.test_set is not None and len(self.test_set) > 0:
                    test_loss, test_correct, test_accuracy = self.test()
                self.epoch += 1

                after_epoch_data: Dict[str, Dict[str, float]] = {
                    'train': {
                        'loss': train_loss
                    },
                    'test': {
                        'loss': test_loss,
                        'correct': test_correct,
                        'accuracy': test_accuracy
                    }
                }

                if self.after_epoch_func is not None:
                    self.after_epoch_func(after_epoch_data)

                if self.epoch % self.save_every_nth_epoch == 0:
                    self.save_network()

                if test_accuracy is not None and test_accuracy > self.accuracy_stop:
                    break
                if max_epochs is not None:
                    if self.epoch == max_epochs:
                        break

            self.save_network()

    def single_train_iteration(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, torch.Tensor]:
        self.optimizer.zero_grad()

        if self.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = self.net(inputs)

        loss = self.net.loss(outputs, targets)
        loss.backward()
        self.optimizer.step()

        return loss.item(), outputs

    def train_iterations(self) -> float:
        train_loss: float = 0
        after_func_num_called: int = 1
        num_iters: int = 0
        for inputs, targets in self.train_loader:
            iteration_loss, outputs = self.single_train_iteration(inputs, targets)
            train_loss += iteration_loss
            num_iters += 1

            if after_func_num_called > 0 and self.after_train_func is not None:
                self.after_train_func(inputs, targets, outputs)
                after_func_num_called -= 1

        return train_loss if num_iters == 0 else train_loss / num_iters

    def scheduler_step(self, loss: float) -> None:
        self.scheduler.step(loss)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def print_train_outcome(self, loss: float) -> None:
        if self.print_messages:
            lr = self.get_lr()
            print('[Epoch: %d] loss: %.10f, LR: %f' % (self.epoch, loss, lr))

    def train(self) -> float:
        self.net.train(True)
        total_loss: float = self.train_iterations()

        self.scheduler_step(total_loss)

        self.print_train_outcome(total_loss)

        return total_loss

    @torch.no_grad()
    def single_test_iteration(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, int, torch.Tensor, int]:
        if self.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs: torch.Tensor = self.net(inputs)

        batch_loss: float = 0
        batch_correct: int = 0

        examples: int = 0

        for i in range(0, outputs.shape[0]):
            target: torch.Tensor = targets[i]
            output: torch.Tensor = outputs[i]

            loss = self.net.loss(output, target)
            batch_loss += loss.item() / outputs.shape[0]

            if self.is_example_correct_func is not None:
                if self.is_example_correct_func(output, target):
                    batch_correct += 1
            else:
                if loss.item() < self.accuracy_threshold:
                    batch_correct += 1

            examples += 1

        return batch_loss, batch_correct, outputs, examples

    @torch.no_grad()
    def test_iterations(self) -> Tuple[float, int, int]:
        test_loss: float = 0
        correct: int = 0
        after_func_num_called: int = 1
        num_iters: int = 0
        total_examples: int = 0
        for inputs, targets in self.test_loader:
            iteration_loss, iteration_correct, outputs, examples = self.single_test_iteration(inputs, targets)
            test_loss += iteration_loss
            correct += iteration_correct
            num_iters += 1
            total_examples += examples

            if after_func_num_called > 0 and self.after_test_func is not None:
                self.after_test_func(inputs, targets, outputs)
                after_func_num_called -= 1

        return test_loss if num_iters == 0 else test_loss / num_iters, correct, total_examples

    @torch.no_grad()
    def print_test_outcome(self, loss: float, correct: int, accuracy: float) -> None:
        if self.print_messages:
            print('[Epoch: %d] Accuracy of the network on the test examples: %.3f %%, loss: %f' %
                  (self. epoch, 100 * accuracy, loss))

    @torch.no_grad()
    def test(self) -> Tuple[float, float, float]:
        self.net.train(False)

        total_loss, total_correct, total_examples = self.test_iterations()
        accuracy: float = float(total_correct / total_examples)

        self.print_test_outcome(total_loss, total_correct, accuracy)

        return total_loss, total_correct, accuracy


class LSTMTrainer(BaseTrainer):
    def scheduler_step(self, loss: float) -> None:
        return

    def single_train_iteration(self, inputs, targets) -> Tuple[float, torch.Tensor]:
        self.optimizer.zero_grad()

        output_size = [s for s in self.net(inputs[0]).shape]
        output_size.insert(0, len(targets))
        outputs = torch.zeros(output_size)
        for n, (inp, target) in enumerate(zip(inputs, targets)):
            outputs[n] = self.net(inp)
        targets = torch.stack(targets)

        loss = self.net.loss(outputs, targets)
        loss.backward()
        self.optimizer.step()

        return loss.item(), outputs

    @torch.no_grad()
    def single_test_iteration(self, inputs, targets) -> Tuple[float, int, torch.Tensor, int]:
        output_size = [s for s in self.net(inputs[0]).shape]
        output_size.insert(0, len(targets))
        outputs = torch.zeros(output_size)

        for n, (inp, target) in enumerate(zip(inputs, targets)):
            outputs[n] = self.net(inp)
        targets = torch.stack(targets)

        batch_loss: float = 0
        batch_correct: int = 0

        examples: int = 0

        for i in range(0, outputs.shape[0]):
            target: torch.Tensor = targets[i]
            output: torch.Tensor = outputs[i]

            loss = self.net.loss(output, target)
            batch_loss += loss.item() / outputs.shape[0]

            if self.is_example_correct_func is not None:
                if self.is_example_correct_func(output, target):
                    batch_correct += 1
            else:
                if loss.item() < self.accuracy_threshold:
                    batch_correct += 1

            examples += 1

        return batch_loss, batch_correct, outputs, examples
