import os
import torch
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from torchvision import transforms

dirname = os.path.dirname(__file__)


def read_data_files(dataset_name, split):
    assert dataset_name in {'MNIST', 'FashionMNIST'}
    assert split in {'train', 'test'}

    folder_path = os.path.join(dirname, '..', 'data', dataset_name, 'raw')
    file_prefix = 't10k' if split == 'test' else split
    image_file = f'{file_prefix}-images-idx3-ubyte'
    label_file = f'{file_prefix}-labels-idx1-ubyte'

    with open(os.path.join(folder_path, image_file), 'rb') as f:
        xs = np.array(np.frombuffer(f.read(), np.uint8, offset=16))

    with open(os.path.join(folder_path, label_file), 'rb') as f:
        ys = np.array(np.frombuffer(f.read(), np.uint8, offset=8))

    xs = np.reshape(xs, (-1, 28, 28, 1)).astype(np.float32)
    ys = ys.astype(np.int)

    return xs, ys


class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, training=True, transform=None, regular=False, fashion=False):
        """
        Create the class for MNIST datasets.

        :param training: bool, whether this is the training or test set
        :param transform: torchvision transforms, what transformations to apply to the images
        :param regular: bool, whether to load the regular MNIST dataset
        :param fashion: bool, whether to load the fashion MNIST dataset
        """
        split = 'train' if training else 'test'
        # If testing
        if regular and fashion:
            # Get the test images and labels for MNIST and FashionMNIST
            x_regular, y_regular = read_data_files('MNIST', split)
            x_fashion, y_fashion = read_data_files('FashionMNIST', split)

            # Join both datasets together
            xs = np.concatenate([x_regular, x_fashion], axis=0)
            ys = np.concatenate([y_regular, y_fashion], axis=0)

            # Shuffle the image and label arrays, keep the same seed for now
            xs, ys = shuffle(xs, ys, random_state=0)

        elif regular:
            xs, ys = read_data_files('MNIST', split)
        elif fashion:
            xs, ys = read_data_files('FashionMNIST', split)
        else:
            raise ValueError('One of regular or fashion MNIST must be selected.')

        self.x_data = xs
        self.y_data = ys
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = Image.fromarray(self.x_data[idx].reshape(28, 28))
        y = torch.tensor(np.array(self.y_data[idx]))
        if self.transform:
            x = self.transform(x)
        x = transforms.ToTensor()(np.array(x)/255)
        return x, y

