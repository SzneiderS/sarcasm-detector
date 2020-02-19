import torch
import torch.nn as nn

from abc import ABC


class BaseModule(nn.Module, ABC):
    def __init__(self):
        super(BaseModule, self).__init__()

    def save(self, filename: str) -> None:
        """Save network to a specified file.

        Network is saved to file with a .net extension.
        Uses PyTorch :func:`torch.save` function to save object to file.

        Args:
            filename (str): File name to save network to (without extension).

        Returns: None

        """
        to_save = self
        torch.save(to_save, filename + ".net")

    @staticmethod
    def load(filename: str):
        """Load network from specified file.

        Network is loaded from file with a .net extension.
        Uses PyTorch :func:`torch.load` function to load object from file.

        Args:
            filename (str): File name to read network from (without extension).

        Returns: Loaded network or None if errors occured.

        """
        try:
            model = torch.load(filename + ".net")
            model.eval()
        except RuntimeError as e:
            print(e)
            return None
        except FileNotFoundError:
            print("File not found. Can't load network")
            return None
        return model

    def load_state(self, filename: str) -> bool:
        """Load network from specified file.

        Network is loaded from file with a .net extension.
        Uses PyTorch :func:`torch.load_state_dict` function to load object state from file.

        Args:
            filename (str): File name to read network from (without extension).

        Returns: Boolean value indicating whether the network has been loaded successfully.

        """
        try:
            model = torch.load(filename + ".net")
            self.load_state_dict(model.state_dict())
            self.eval()
        except RuntimeError as e:
            print(e)
            return False
        except FileNotFoundError:
            print("File not found. Can't load network")
            return False
        return True
