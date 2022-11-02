import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class FlipHorizontal(Transform):
    def __init__(self):
        pass

    def __call__(self, img):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, _x):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Sampler:
    """Base class for all Samplers.
    Every Sampler subclass has to provide an `__iter__` method, providing a
    way to iterate over indices of dataset elements, and a `__len__` method
    that returns the length of the returned iterators.
    """

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise TypeError


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.
    Args:
        data_source (Dataset): dataset to sample from
    """

    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __iter__(self) -> Iterator[int]:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify `num_samples` to draw.
    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """
    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise TypeError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(self.replacement)
            )

        if self.num_samples is not None and not replacement:
            raise ValueError(
                "With replacement=False, num_samples should not be specified, "
                "since a random permute will be performed."
            )

        if self.num_samples is None:
            self.num_samples = len(data_source)
        # if not isinstance(self.num_samples, int) or self.num_samples <= 0:
        #     raise ValueError("num_samples should be a positive integer "
        #                      "value, but got num_samples={}".format(self.num_samples))

    def __iter__(self) -> Iterator[int]:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self, sampler: Union[Sampler, Iterable], batch_size: int, drop_last: bool
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def default_collate():
    r"""Puts each data field into a tensor with outer dimension batch size"""
    raise NotImplementedError


def collate_mnist(batch):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, p: Optional[int] = 0.5, transforms: Optional[List] = None):
        self.p = p
        self.transforms = transforms
        self.perform_transforms = transforms is not None

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise TypeError

    def apply_transforms(self, x):
        if self.perform_transforms:
            if np.random.rand() < self.p:
                # apply the transforms
                for tform in self.transforms:
                    x = tform(x)
        return x


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
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
    """
    dataset: Dataset
    batch_size: Optional[int]
    drop_last: bool
    sampler: Sampler
    _iterator: Optional["_BaseDataLoaderIter"]
    __initialized = False

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Union[Sampler, Iterable, None] = None,
        collate_fn: Optional = default_collate,
        drop_last: bool = False,
    ):

        self.dataset = dataset
        self.collate_fn = collate_fn

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with " "shuffle")

        if sampler is None:  # give default samplers
            # We are only doing map style datasets
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler

        if batch_size > 1:
            self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last)

        self.collate_fn = collate_fn

        self.__initialized = True
        self._IterableDataset_len_called = None

        self._iterator = None

    def _get_iterator(self) -> "_BaseDataLoaderIter":
        return _SingleProcessDataLoaderIter(self)

    # We quote '_BaseDataLoaderIter' since it isn't defined yet and the definition can't be moved up
    # since '_BaseDataLoaderIter' references 'DataLoader'.
    def __iter__(self) -> "_BaseDataLoaderIter":
        # When using a single worker the returned iterator should be
        # created everytime to avoid reseting its state
        # However, in the case of a multiple workers iterator
        # the iterator is only created once in the lifetime of the
        # DataLoader object so that workers can be reused
        return self._get_iterator()

    @property
    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
        if self.batch_size > 1:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self) -> int:
        length = self._IterableDataset_len_called = len(self.dataset)
        if self.batch_size is not None:
            from math import ceil

            if self.drop_last:
                length = length // self.batch_size
            else:
                length = ceil(length / self.batch_size)
        return length


class _BaseDataLoaderIter(object):
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self._dataset = loader.dataset
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._sampler_iter = iter(self._index_sampler)
        self._collate_fn = loader.collate_fn
        self._base_seed = np.empty((), dtype=np.int64)
        self._num_yielded = 0
        self._profile_name = "enumerate(DataLoader)#{}.__next__".format(
            self.__class__.__name__
        )

    def __iter__(self) -> "_BaseDataLoaderIter":
        return self

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        if self._sampler_iter is None:
            self._reset(self.loader)
        data = self._next_data()
        self._num_yielded += 1
        return data

    next = __next__  # Python 2 compatibility

    def __len__(self) -> int:
        return len(self._index_sampler)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)

        self._dataset_fetcher = _IterableDatasetFetcher(
            self._dataset, self._collate_fn, self._drop_last
        )

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        return data


class _BaseDatasetFetcher(object):
    def __init__(self, dataset, collate_fn, drop_last):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, collate_fn, drop_last):
        super(_IterableDatasetFetcher, self).__init__(dataset, collate_fn, drop_last)
        self.dataset = dataset
        self.ended = False

    def fetch(self, possibly_batched_index):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
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
            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filesname: str,
        label_filename: str,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
