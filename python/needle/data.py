import numpy as np
from .autograd import Tensor
import os
import struct, gzip, pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd

from tqdm import tqdm, trange


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        return np.flip(img, axis=-2) if flip_img else img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        pad_width = ((self.padding, self.padding),
                     (self.padding, self.padding),
                     (0, 0))
        if img.ndim == 4:
            pad_width = ((0, 0),) + pad_width
        return np.pad(img,
                      pad_width,
                      mode='constant', constant_values=0)[..., self.padding + shift_x :
                                                               self.padding + shift_x + img.shape[-3],
                                                               self.padding + shift_y :
                                                               self.padding + shift_y + img.shape[-2], :]
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
        self.h = self.w = 28
        self.transforms = transforms
        self.transforms_fn = lambda I : np.reshape(self.apply_transforms(np.reshape(I,
                                                                                    (-1, self.h, self.w))[..., None]),
                                                   (-1, self.h * self.w))
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        try:
            img, lbl = self.img[index], self.lbl[index] # image shape: (B, H x W)
        except AttributeError:
            self.img, self.lbl = parse_mnist(self.image_filename, self.label_filename)
            return self.transforms_fn(self.img[index]), self.lbl[index]
        else:
            return self.transforms_fn(img), lbl
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        try:
            return len(self.lbl)
        except AttributeError:
            self.img, self.lbl = parse_mnist(self.image_filename, self.label_filename)
            return len(self.lbl)
        ### END YOUR SOLUTION


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train       - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - NumPy array of images
        y - NumPy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.train = train
        if train:
            self.img = np.empty(shape=(50000, 3072),
                                dtype=np.uint8)
            self.lbl = np.empty(shape=(50000,),
                                dtype=np.float32)
            # Training data are packed into 5 files.
            # see https://www.cs.toronto.edu/~kriz/cifar.html
            for i in range(5):
                with open(os.path.join(base_folder, f'data_batch_{i+1}'), 'rb') as f:
                    training_batch = pickle.load(f, encoding='bytes')
                self.img[i * 10000 : (i+1) * 10000] = training_batch[b'data']
                self.lbl[i * 10000 : (i+1) * 10000] = np.array(training_batch[b'labels'],
                                                               dtype=np.float32)
        else:
            with open(os.path.join(base_folder, 'test_batch'), 'rb') as f:
                test_set = pickle.load(f, encoding='bytes')
            self.img = test_set[b'data']
            self.lbl = np.array(test_set[b'labels'],
                                dtype=np.float32)
        self.transforms = transforms
        self.transforms_fn = lambda I, singleton : \
                                    self.apply_transforms(I.astype(np.float32).reshape((3, 32, 32) if   singleton \
                                                                                                   else (-1, 3, 32, 32)) / 255.)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        return self.transforms_fn(self.img[index],
                                  isinstance(index, int)), self.lbl[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return 50000 if self.train else 10000
        ### END YOUR SOLUTION


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
            # see https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

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






class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test  = self.tokenize(os.path.join(base_dir, 'test.txt' ), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path      - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        assert self.dictionary.add_word('<eos>') == 0, "'<eso>' hardwired to ID 0"

        def add_tokens(line, ids):
            # `line` is guaranteed to be non-empty.
            for word in line.split():
                ids.append(self.dictionary.add_word(word))
            ids.append(0)
        
        ids = []
        with open(path) as f:
            # The effects of `max_lines` being 0 and `None` are equivalent.
            # Both of them are considered FALSE.
            if max_lines:
                for _ in trange(max_lines):
                    line = f.readline()
                    if not line:
                        break
                    add_tokens(line, ids)
            else:
                line = f.readline()
                # If `max_lines` exceeds the number of lines in the file,
                # `readline` will return an empty string.
                while line:
                    add_tokens(line, ids)
                    line = f.readline()
        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, `batchify` arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means, e.g.,
    that the dependence of 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a NumPy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    # see https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
    get_chunks = lambda data, batch_size: np.array_split(data,
                                                         batch_size)
    size = len(data)
    trim = size % batch_size
    # `device` not applicable to NumPy calls.
    return np.array(list(get_chunks(data[ : size - trim],
                                    batch_size)),
                         dtype=dtype).T
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    subdivides the source data into chunks of length `bptt`
    Given a bptt-limit 2 and the above output of `batchify`, we will get the
    following two variables when `i` is 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Inputs:
    batches - NumPy array returned from batchify function
    i       - index
    bptt    - sequence length
    Returns:
    data   - tensor of shape (`bptt`       , `bs`) with NDArray as `cached_data`
    target - tensor of shape (`bptt` x `bs`,     ) with NDArray as `cached_data`
             `target`, or ground truth against the predictions, is flattened
             to fit the parameter `y` of `SoftmaxLoss`.
    """
    ### BEGIN YOUR SOLUTION
    return Tensor(array=batches[i : i + bptt],
                  device=device,
                  dtype=dtype,
                  requires_grad=False), \
           Tensor(array=batches[i + 1 : i + 1 + bptt].flatten(),
                  device=device,
                  dtype=dtype,
                  requires_grad=False)
    ### END YOUR SOLUTION