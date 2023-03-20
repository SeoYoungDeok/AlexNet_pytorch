from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


class MnistDataLoader(DataLoader):
    def __init__(self, path, batch, shuffle=True):
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(256),
                transforms.RandomCrop(227),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        os.makedirs(path, exist_ok=True)

        self.dataset = datasets.MNIST(
            path, train=True, download=True, transform=transform
        )
        super().__init__(self.dataset, batch_size=batch, shuffle=shuffle)
