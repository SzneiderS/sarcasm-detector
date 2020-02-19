from training.trainers import LSTMTrainer
from models.base import BaseModule
from models.datasets.headlines_dataset import HeadlinesDataset

from utils import calculate_one_hot_from_file

from torchvision.transforms import transforms

import torch
import torch.nn as nn
from samplers import EqualClassSampler

from visualization import TrainerVisualizer

import random


class HeadlineClassifier(BaseModule):
    def __init__(self, embedding_size, hidden_size=2, recurrent_layers=1):
        super(HeadlineClassifier, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.recurrent_layers = recurrent_layers

        self.features_size = int(embedding_size * 0.5)

        self.embedding2features = nn.Linear(embedding_size, self.features_size)

        self.lstm = nn.LSTM(self.features_size, self.hidden_size, batch_first=True, num_layers=recurrent_layers)

        self.lstm2out = nn.Sequential(
            nn.Linear(self.hidden_size, 30),
            nn.PReLU(),
            nn.Linear(30, 15),
            nn.PReLU(),
            nn.Linear(15, 2)
        )

        self.dropout = nn.Dropout(0.1)

        self.loss_function = nn.CrossEntropyLoss()

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, x):
        if self.is_cuda:
            x = x.cuda()

        features = torch.zeros(x.size(0), self.features_size)
        if self.is_cuda:
            features = features.cuda()
        for n, i in enumerate(x):
            features[n] = self.embedding2features(i)

        x = features

        out, _ = self.lstm(x.unsqueeze(0))

        x = out.view(-1, self.hidden_size)[-1]

        x = self.dropout(x)
        x = self.lstm2out(x)
        return x

    def predict(self, x):
        return torch.argmax(self.forward(x))

    def loss(self, output, target):
        if self.is_cuda:
            output, target = output.cuda(), target.cuda()
        if self.training:
            loss = self.loss_function(output, target.view(output.size(0)))
        else:
            loss = self.loss_function(output.view(1, output.size(0)), target)
        sum_params = torch.zeros(1)
        if self.is_cuda:
            sum_params = sum_params.cuda()
        for p in self.parameters():
            sum_params += p.pow(2).sum()  # regularization
        return loss + sum_params * 1e-5


def onehot_char():
    def preprocess_file(file):
        import json

        content = file.read().split('\n')

        classes_set = set()

        for line in content:
            data = json.loads(line)
            for c in data["headline"]:
                classes_set.add(c)
        return classes_set

    vectors, vector_length = calculate_one_hot_from_file("data/headlines.json", preprocess_file)

    def postprocess_data(example):
        headline, sarcasm = example
        processed_headline = torch.zeros(len(headline), vector_length)
        for n, c in enumerate(headline):
            processed_headline[n] = torch.Tensor(vectors[c])
        return processed_headline, sarcasm

    return vector_length, postprocess_data


def embeddings(embedding_size=100, pretrained_file=None):
    from gensim.models import Word2Vec, KeyedVectors
    from nltk.stem import WordNetLemmatizer
    import nltk
    from nltk.corpus import stopwords
    import re
    import json

    nltk.download('wordnet')
    nltk.download('stopwords')

    wnl = WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))

    def process_word(word):
        word = wnl.lemmatize(word)
        return word

    class SentencesIter:
        def __init__(self, file):
            self.content = file.read().split('\n')
            self.current = 0

        def __iter__(self):
            return self

        def reset(self):
            self.current = 0

        def __next__(self):
            if self.current >= len(self.content):
                self.reset()
                raise StopIteration
            line = self.content[self.current]
            data = json.loads(line)
            words = re.findall(r"[a-z0-9]+'?[a-z]?", data["headline"].lower())
            words = [process_word(word) for word in words if word not in stopwords_set]
            self.current += 1
            return words

    model = None
    if pretrained_file is not None:
        model = KeyedVectors.load_word2vec_format(pretrained_file)
        embedding_size = model.vector_size
    else:
        with open("data/headlines.json", "r") as f:
            sentences = SentencesIter(f)
            model = Word2Vec(sentences, iter=10, compute_loss=True, size=embedding_size)
            training_iters = 0
            while True:
                prev_loss = model.get_latest_training_loss()
                print("Word2vec training loss: {}".format(model.get_latest_training_loss()))
                model.train(sentences, compute_loss=True, total_examples=model.corpus_count, epochs=10)
                if (prev_loss - model.get_latest_training_loss()) <= 0:
                    training_iters += 1
                    if training_iters >= 25:
                        break

    def postprocess_data(example):
        headline, sarcasm = example
        words = re.findall(r"[a-z0-9]+'?[a-z]?", headline.lower())
        words = [process_word(word) for word in words if word not in stopwords_set]
        in_vocab = [model.wv[word] for word in words if word in model.wv]
        if len(in_vocab) == 0:
            raise ValueError("Empty vocabulary for sentence '{}'".format(headline))
        if len(in_vocab) == 1:
            raise ValueError("Processed sentence ('{}') has only one word embedding found".format(headline))
        processed_headline = torch.zeros(len(in_vocab), embedding_size)
        for n, word in enumerate(in_vocab):
            processed_headline[n] = torch.FloatTensor(word)
        return processed_headline, sarcasm

    return embedding_size, postprocess_data


if __name__ == "__main__":
    random.seed(2020)
    limit = 10000

    # vector_length, postprocess_func = onehot_char()
    vector_length, postprocess_func = embeddings(50)
    # vector_length, postprocess_func = embeddings(pretrained_file="data/glove/tweets100d.txt")

    transforms = {
        "input": transforms.Lambda(lambda x: x.view(x.size(0), vector_length)),
        "target": transforms.Lambda(lambda x: torch.LongTensor([x]))
    }

    train_set, test_set = HeadlinesDataset.from_json("data/headlines.json", postprocess_data_func=postprocess_func,
                                                     shuffle_dataset=True, transforms=transforms, limit=limit)

    batch_size = 256

    train_sampler = EqualClassSampler(train_set, shuffle=True)
    test_sampler = EqualClassSampler(test_set)

    net = HeadlineClassifier(vector_length, 5)
    trainer = LSTMTrainer(net, train_set, test_set, batch_size=batch_size, lr=1e-2, train_sampler=train_sampler,
                          test_sampler=test_sampler)

    def collate(batch):
        inputs = [i[0] for i in batch]
        targets = [i[1] for i in batch]
        return inputs, targets

    trainer.train_loader.collate_fn = collate
    trainer.test_loader.collate_fn = collate

    # trainer.overfit_batch_loss = 0.01

    def example_correct(output, target):
        return torch.argmax(output) == target
    trainer.is_example_correct_func = example_correct

    visualizer = TrainerVisualizer()
    trainer.after_epoch_func = visualizer.update_scores

    trainer.run(max_epochs=150)
    visualizer.save("plot.png")
