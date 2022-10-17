import struct, gzip
import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """ Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C NDArray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        return np.flip(img, axis=1) if flip_img else img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding,
                                             high=self.padding+1,
                                             size=2)
        ### BEGIN YOUR SOLUTION
        if img.ndim == 3: # RGB image: do not pad channel dimension
            return np.pad(img,
                          ((self.padding, self.padding),
                           (self.padding, self.padding),
                           (0, 0)),
                          mode='constant', constant_values=0)[self.padding + shift_x :
                                                              self.padding + shift_x + img.shape[0],
                                                              self.padding + shift_y :
                                                              self.padding + shift_y + img.shape[1], :]
        elif img.ndim == 2: # grayscale image
            return np.pad(img,
                          ((self.padding, self.padding),
                           (self.padding, self.padding)),
                          mode='constant', constant_values=0)[self.padding + shift_x :
                                                              self.padding + shift_x + img.shape[0],
                                                              self.padding + shift_y :
                                                              self.padding + shift_y + img.shape[1]]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # decompression
    # reference: https://docs.python.org/3/library/gzip.html#gzip.open
    with gzip.open(image_filename, 'rb') as img_obj:
        img = img_obj.read()
    with gzip.open(label_filename, 'rb') as lbl_obj:
        lbl = lbl_obj.read()
    # determine dataset size
    # reference: https://blog.csdn.net/lindorx/article/details/94639183
    num_img, H, W = struct.unpack_from('>iii', # big endian
                                       img,
                                       offset=4)
    num_lbl = struct.unpack_from('>i', # big endian
                                 lbl,
                                 offset=4)
    # `num_lbl` is a tuple.
    assert num_img == num_lbl[0], '# of images and labels do not match'
    # load data to NumPy
    img = struct.unpack_from('>' + 'B' * H * W * num_img, # big endian
                             img,
                             offset=16)
    img = np.array(img, dtype=np.float32).reshape(num_img, H * W)
    img = np.clip(img / 255. ,
                  0., 1.)
    lbl = struct.unpack_from('>' + 'B' * num_lbl[0], # big endian
                             lbl,
                             offset=8)
    lbl = np.array(lbl, dtype=np.uint8)

    return img, lbl
    ### END YOUR CODE

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.image_filename = image_filename
        self.label_filename = label_filename
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        h = w = 28
        tform = lambda I : np.reshape(self.apply_transforms(np.reshape(I, (-1,h,w))),
                                      (-1, h*w))
        try:
            return tform(self.img[index]), self.lbl[index]
        except AttributeError:
            self.img, self.lbl = parse_mnist(self.image_filename, self.label_filename)
            return tform(self.img[index]), self.lbl[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        try:
            return len(self.lbl)
        except AttributeError:
            self.img, self.lbl = parse_mnist(self.image_filename, self.label_filename)
            return len(self.lbl)
        
        ### END YOUR SOLUTION


# reference: https://stackoverflow.com/questions/62549990/what-does-next-and-iter-do-in-pytorchs-dataloader
class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            # reference: https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size,
                                                 len(dataset),
                                                 batch_size))
    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            # reference: https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html
            ordering = np.random.permutation(len(self.dataset))
            self.ordering = np.array_split(ordering,
                                           range(self.batch_size,
                                                 len(ordering),
                                                 self.batch_size))
        self.num_iter = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        tensorify = lambda X : Tensor(X, dtype=X.dtype,
                                         requires_grad=False)
        try:
            batch_idx = self.ordering[self.num_iter]
        except IndexError:
            raise StopIteration
        else:
            self.num_iter += 1
            batch = self.dataset[batch_idx]
            return tuple(map(tensorify, batch))
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
