import _infimnist as infimnist
import numpy as np
import ngclearn.utils.transform_utils as transform

class InfiMNIST(object):
    """
    The Infinite MNIST sampler.

    Args:
        batch_size: number of samples to place inside a mini-batch
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.mnist_sampler = infimnist.InfimnistGenerator()
        self.ptr = 70000

    def sample(self):
        """
        Draws a mini-batch of samples from the current state of this
        infinite mnist generating process.

        Returns:
            (x, y), where x is a sample of shape (B x 784) and y is
                a (B x 10) one-hot binary vector (B is mini-batch length)
        """
        ptrs = list(range(self.ptr, self.ptr + self.batch_size, 1))
        indices = np.asarray(ptrs, dtype=np.int64)
        self.ptr = self.ptr + self.batch_size # move pointer
        # sample/retrieve the mnist patterns
        digits, labels = self.mnist_sampler.gen(indices) # get images and labels
        # post-process the sampled patterns
        x = digits.astype(np.float32).reshape(indices.shape[0], 28, 28)
        x = x / 255 # normalize
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2]) # flatten
        y = transform.to_one_hot(labels,depth=10) # convert labels to binary one-hot
        return x, y
