import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class TrainerVisualizer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(dpi=100, figsize=(16, 8))

        self.losses = self.fig.add_subplot(211)
        self.train_loss_plot, = self.losses.plot([], [], label="Train loss")
        self.test_loss_plot, = self.losses.plot([], [], label="Test loss")
        self.losses.legend()
        self.losses.set_title("Train & test loss")
        self.losses.set_xlabel("Epochs")
        self.losses.set_ylabel("Loss")
        self.losses.xaxis.set_major_locator(MaxNLocator(integer=True))

        self.accuracy = self.fig.add_subplot(212)
        self.accuracy_plot, = self.accuracy.plot([], [], label="Accuracy")
        self.accuracy.set_title("Test set accuracy")
        self.accuracy.set_xlabel("Epochs")
        self.accuracy.set_ylabel("Accuracy (%)")
        self.accuracy.xaxis.set_major_locator(MaxNLocator(integer=True))

        self.train_losses, self.test_losses = [], []
        self.accuracies = []

    def update_scores(self, scores):
        self.train_losses.append(scores["train"]["loss"])
        self.test_losses.append(scores["test"]["loss"])
        self.accuracies.append(scores["test"]["accuracy"] * 100)

        if len(self.train_losses) > 1:
            self.update_plots()

    def update_plots(self):
        epochs = [train + 1 for train in range(len(self.train_losses))]

        self.test_loss_plot.set_data(epochs, self.test_losses)
        self.train_loss_plot.set_data(epochs, self.train_losses)
        self.accuracy_plot.set_data(epochs, self.accuracies)

        self.losses.relim()
        self.losses.autoscale_view()
        self.accuracy.relim()
        self.accuracy.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.tight_layout()

    def save(self, filename):
        self.fig.savefig(filename)
