# %%
import math
import numpy as np
from multiprocessing import Value

class MaskCollator(object): # Adapted from the Meta JEPA code-base to ngc-learn compliance
    """
    A mechanism for generating/creating patch masks, generally for self-supervised learning.

    Args:
        cfgs_mask: configuration masks to apply

        crop_size: dimensions of crop

        patch_size:  dimensions of patches to create
    """

    def __init__(self, cfgs_mask, crop_size=(224, 224), patch_size=(16, 16),):
        super(MaskCollator, self).__init__()

        self.mask_generators = []
        for m in cfgs_mask:
            mask_generator = _MaskGenerator(
                crop_size=crop_size,
                patch_size=patch_size,
                pred_mask_scale=m.get('spatial_scale'),
                aspect_ratio=m.get('aspect_ratio'),
                npred=m.get('num_blocks'),
                max_keep=m.get('max_keep', None),
            )
            self.mask_generators.append(mask_generator)

    def step(self):
        """
         Steps this generator forward one step.

         Returns:
             next set of collated encoder masks, next set of predictor masks
         """
        for mask_generator in self.mask_generators:
            mask_generator.step()

    def __call__(self, batch):
        batch_size = len(batch)
        collated_masks_pred, collated_masks_enc = [], []
        for i, mask_generator in enumerate(self.mask_generators):
            masks_enc, masks_pred = mask_generator(batch_size)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)

        return collated_masks_enc, collated_masks_pred


class _MaskGenerator(object):

    def __init__(
            self,crop_size=(224, 224), patch_size=(16, 16), pred_mask_scale=(0.2, 0.8), aspect_ratio=(0.3, 3.0),
            npred=1,max_keep=None
    ):
        super(_MaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, ) * 2
        self.crop_size = crop_size
        self.height, self.width = crop_size[0] // patch_size[0], crop_size[1] // patch_size[1]

        self.patch_size = patch_size
        self.aspect_ratio = aspect_ratio
        self.pred_mask_scale = pred_mask_scale
        self.npred = npred
        self.max_keep = max_keep
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value = (i.value + 1) % 2**16
            v = i.value
        return v

    def _sample_block_size(self,rng: np.random.RandomState,scale, aspect_ratio_scale):
        # -- Sample spatial block mask scale
        _rand = rng.random()
        min_s, max_s = scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # -- Sample block aspect-ratio
        _rand = rng.random()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (h, w)

    def _sample_block_mask(self, b_size, rng: np.random.RandomState):
        h, w = b_size
        top = rng.randint(0, self.height - h + 1)
        left = rng.randint(0, self.width - w + 1)

        mask = np.ones((self.height, self.width), dtype=np.int32)
        mask[top:top+h, left:left+w] = 0

        return mask

    def __call__(self, batch_size):
        """
        Create encoder and predictor masks when collating imgs into a batch:

        | # 1. sample pred block size using seed
        | # 2. sample several pred block locations for each image (w/o seed)
        | # 3. return pred masks and complement (enc mask)

        Args:
            batch_size: number of samples to place w/in a generate batch

        Returns:
            collated encoder masks, collated predictor masks
        """
        seed = self.step()
        rng = np.random.RandomState(seed)
        p_size = self._sample_block_size(rng=rng, scale=self.pred_mask_scale, aspect_ratio_scale=self.aspect_ratio,)

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.height * self.width
        for _ in range(batch_size):
            empty_context = True
            while empty_context:
                # Create a mask for this sample
                mask_e = np.ones((self.height, self.width), dtype=np.int32)
                for _ in range(self.npred):
                  mask_e *= self._sample_block_mask(p_size, rng)
                mask_e = mask_e.flatten()

                mask_p = np.where(mask_e == 0)[0]
                mask_e = np.where(mask_e != 0)[0]

                empty_context = len(mask_e) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        # Truncate arrays to the minimum length to create uniform arrays
        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        collated_masks_pred = np.array(collated_masks_pred)

        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_enc = np.array(collated_masks_enc)

        return collated_masks_enc, collated_masks_pred
