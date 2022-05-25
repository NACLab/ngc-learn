"""
This is test/driver for probing if the infinite mnist sampler
works properly as a data generating process.

Should produce a set of 10 images saved to disk with the correct binary one-hot
label in the produced "title" bar of each image.
"""

from ngclearn.generator.experimental.infimnist.infimnist import InfiMNIST
import matplotlib.pyplot as plt

n_iter = 10

sampler = InfiMNIST(batch_size=8)

for t in range(n_iter):
    x, y = sampler.sample()
    print(x.shape)
    print(y.shape)

    plt.imshow(x[0].reshape(28,28))
    plt.title('label: {}'.format(y[0]))
    plt.savefig("sample{}.jpg".format(t))
