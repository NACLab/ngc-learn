"""
Image/tensor patching utility routines.
"""
import numpy as np
from jax import numpy as jnp
from sklearn.feature_extraction.image import extract_patches_2d


def patch_with_stride(X, patch, stride):
    n_x, n_y = X.shape[0] // patch[0], X.shape[1] // patch[1]
    result = []
    for x_idx in range(n_x):
        for y_idx in range(n_y):

            x_start = stride[0] * x_idx
            x_end = x_start + patch[0]
            y_start = stride[1] * y_idx
            y_end = y_start + patch[1]

            img_ = X[x_start: x_end, y_start: y_end]

            if img_.shape == (patch[0], patch[1]):
                result.append(img_)

    return jnp.array(result)


def patch_with_overlap(X, patch, overlap):
    
    stride = (X.shape[0] - overlap[0],
              X.shape[1] - overlap[1])
    
    return patch_with_stride(X, patch, stride)
  

class Create_Patches:
    """
    This function will create small patches out of the image based on the provided attributes.

    Args:
        img: jax array of size (H, W)

        patch_shape: (height_patch, width_patch)

        overlap_shape: (height_overlap, width_overlap)

    Returns:
        jnp.array: Array containing the patches, shape: (num_patches, patch_height, patch_width)

    """
    #patched: (height_patch, width_patch)
    #overlap: (height_overlap, width_overlap)
    #add_frame: increases the img size by (height_patch - height_overlap, width_patch - width_overlap)
    #create_patches: creates small patches out of the image based on the provided attributes.

    def __init__(self, img, patch_shape, overlap_shape):
        self.img = img
        self.height_patch, self.width_patch = patch_shape
        self.height_over, self.width_over = overlap_shape

        self.height, self.width = self.img.shape

        self.nw_patches = self.width // self.width_patch #(self.width + self.width_over) // (self.width_patch - self.width_over) - 2
        self.nh_patches = self.height // self.height_patch #(self.height + self.height_over) // (self.height_patch - self.height_over) - 2

    def _add_frame(self):
        """
        This function will add zero frames (increase the dimension) to the image

        Returns:
            image with increased size (x.shape[0], x.shape[1]) -> (x.shape[0] + (height_patch - height_overlap),
                                                                   x.shape[1] + (width_patch - width_overlap))
        """
        self.img = np.hstack((jnp.zeros((self.img.shape[0], (self.height_patch - self.height_over))),
                              self.img,
                              jnp.zeros((self.img.shape[0], (self.height_patch - self.height_over)))))
        self.img = np.vstack((jnp.zeros(((self.width_patch - self.width_over), self.img.shape[1])),
                              self.img,
                              jnp.zeros(((self.width_patch - self.width_over), self.img.shape[1]))))

        self.height, self.width = self.img.shape

        self.nw_patches = (self.width + self.width_over) // (self.width_patch - self.width_over) - 2
        self.nh_patches = (self.height + self.height_over) // (self.height_patch - self.height_over) - 2



    def create_patches(self, add_frame=False, center=True):
        """
        This function will create small patches out of the image based on the provided attributes.

        Keyword Args:
            add_frame: If true the function will add zero frames (increase the dimension) to the image

            center:

        Returns:
            jnp.array: Array containing the patches
            shape: (num_patches, patch_height, patch_width)
        """

        if add_frame == True:
            self._add_frame()

        if center == True:
            mu = np.mean(self.img, axis=0, keepdims=True)
            self.img  = self.img - mu

        result = []
        for nh_ in range(self.nh_patches):
            for nw_ in range(self.nw_patches):
                img_ = self.img[(self.height_patch - self.height_over) * nh_: nh_ * (
                            self.height_patch - self.height_over) + self.height_patch
                , (self.width_patch - self.width_over) * nw_: nw_ * (
                            self.width_patch - self.width_over) + self.width_patch]

                if img_.shape == (self.height_patch, self.width_patch):
                    result.append(img_)
        return jnp.array(result)





def generate_patch_set(x_batch, patch_size=(8, 8), max_patches=50, center=True, seed=1234, vis_mode=False): ## scikit
    """
    Generates a set of patches from an array/list of image arrays (via
    random sampling with replacement). This uses scikit-learn's patch creation
    function to generate a set of (px x py) patches.
    Note: this routine also subtracts each patch's mean from itself.

    Args:
        x_batch: the array of image arrays to sample from

        patch_size: a 2-tuple of the form (pH = patch height, pW = patch width)

        max_patches: maximum number of patches to extract/generate from source images

        center: centers each patch by subtracting the patch mean (per-patch)

        seed: seed to control the random state of internal patch sampling

    Returns:
        an array (D x (pH * pW)), where each row is a flattened patch sample
    """
    _x_batch = np.array(x_batch)
    px = py = int(np.sqrt(_x_batch.shape[1])) # get image shape of the data
    p_batch = None
    for s in range(_x_batch.shape[0]):
        xs = _x_batch[s, :]
        xs = xs.reshape(px, py)
        patches = extract_patches_2d(xs, patch_size, max_patches=max_patches, random_state=seed)#, random_state=69)
        patches = np.reshape(patches, (len(patches), -1)) # flatten each patch in set
        if s > 0:
            p_batch = np.concatenate((p_batch,patches),axis=0)
        else:
            p_batch = patches
            
    mu = 0
    if center: ## center patches by subtracting out their means
        mu = np.mean(p_batch, axis=1, keepdims=True)
        p_batch = p_batch - mu
    if vis_mode:
        return jnp.array(p_batch), mu
    else:
        return jnp.array(p_batch)
        

def generate_pacthify_patch_set(x_batch_, patch_size=(5, 5), center=True): ## patchify
    ## this is a patchify-specific function (only use if you have patchify installed...)
    import patchify as ptch
    """
    Generates a set of patches from an array/list of image arrays (via
    random sampling with replacement). This uses the patchify library to create
    a of non-random non-overlapping or overlapping (w/ controllable stride) patches.
    Note: this routine also subtracts each patch's mean from itself.

    Args:
        x_batch_: the array of image arrays to sample from

        patch_size: a 2-tuple of the form (pH = patch height, pW = patch width)

        center: centers each patch by subtracting the patch mean (per-patch)

    Returns:
        an array (D x (pH * pW)), where each row is a flattened patch sample
    """
    x_batch = np.array(x_batch_)
    px = py = int(np.sqrt(x_batch.shape[1])) # get image shape of the data
    x_batch = np.expand_dims(x_batch.reshape(px, py), axis=2)
    pch_x = patch_size[0]
    pch_y = patch_size[1]
    pX = np.squeeze( ptch.patchify(x_batch, (pch_x,pch_y,1), step=pch_x) ) # step = stride
    patchBatch = []
    for i in range(pX.shape[0]):
        for j in range(pX.shape[1]):
            _p = np.reshape(pX[i,j,:,:], (1, pch_x * pch_y))
            patchBatch.append(_p)
    patchBatch = jnp.concatenate(patchBatch, axis=0)
    if center == True:
        mu = np.mean(patchBatch, axis=1,keepdims=True)
        patchBatch = patchBatch - mu
    return patchBatch
