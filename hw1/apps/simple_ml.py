import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
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
    with gzip.open(image_filesname, 'rb') as img_obj:
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
    img = img / 255.
    img = np.clip(img, 0., 1.)
    lbl = struct.unpack_from('>' + 'B' * num_lbl[0], # big endian
                             lbl,
                             offset=8)
    lbl = np.array(lbl, dtype=np.uint8)

    return img, lbl
    ### END YOUR CODE


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    # Needle does not support operation `mean`.
    loss = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,))) - \
           ndl.summation(ndl.multiply(Z, y_one_hot), axes=(1,)) # `axes` must be a tuple.
    return ndl.summation(loss) / Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    ### BEGIN YOUR SOLUTION
    num_iter = int(np.ceil(X.shape[0] / batch))
    for i in range(num_iter):
        # get batch
        if i == num_iter - 1:
            X_batch = X[i * batch : ]
            y_batch = y[i * batch : ]
        else:
            X_batch = X[i * batch : (i + 1) * batch]
            y_batch = y[i * batch : (i + 1) * batch]
        X_batch = ndl.Tensor(X_batch, dtype='float32')
        # forward pass
        Z = ndl.matmul(ndl.relu(ndl.matmul(X_batch, W1)), W2)
        y_one_hot = np.zeros(Z.shape, dtype=np.int8) # This line must follow `Z`.
        y_one_hot[np.arange(y_batch.shape[0]), y_batch] = 1.
        y_one_hot = ndl.Tensor(y_one_hot, dtype='int8')
        loss = softmax_loss(Z, y_one_hot)
        # backprop
        loss.backward()
        # gradient descent
        W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy(), dtype='float32')
        W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy(), dtype='float32')
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
