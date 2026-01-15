from typing import List
import time
import torch
import numpy as np

from .OSNet import osnet_ibn_x1_0
from .utils import load_pretrained_weights
from .image_handler import normalize, resize, ndarray_to_tensor


class OsNetEncoder:

    PRETRAINED_MODEL = False
    LOSS = 'softmax'

    def __init__(
        self,
        input_width: int,
        input_height: int,
        weight_filepath: str,
        batch_size: int,
        num_classes: int,
        patch_height: int,
        patch_width: int,
        norm_mean: List[float],
        norm_std: List[float],
        GPU: bool
    ):
        self._input_width = input_width
        self._input_height = input_height
        self._weight_filepath = weight_filepath
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.patch_size = (patch_height, patch_width)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.GPU = GPU

        self._model = osnet_ibn_x1_0(
            num_classes=self.num_classes,
            loss=OsNetEncoder.LOSS,
            pretrained=OsNetEncoder.PRETRAINED_MODEL,
            use_gpu=self.GPU
        )

        self._model.eval()
        load_pretrained_weights(self._model, self._weight_filepath)

        if self.GPU:
            self._model = self._model.cuda()

    def load_image(self, patch: np.ndarray) -> torch.Tensor:
        device = torch.device("cuda" if self.GPU else "cpu")

        resized = resize(patch, self.patch_size)
        tensor = ndarray_to_tensor(resized)
        tensor = normalize(tensor, self.norm_mean, self.norm_std, inplace=False)

        return tensor.to(device)

    def get_features(self, image_patches: List[np.ndarray]) -> List[np.ndarray]:
        """
        Returns L2-normalized float32 feature vectors (SAFE FOR COSINE)
        """

        features = []

        for patch in image_patches:
            if patch is None:
                features.append(None)
                continue

            start = time.time()

            patch_tensor = self.load_image(patch)

            with torch.no_grad():
                feat = self._model(patch_tensor)

            feat = (
                feat.cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
            )

            norm = np.linalg.norm(feat)
            if norm > 0:
                feat /= norm

            features.append(feat)

            print(f"[PERFORMANCE] Features extracted: {1/(time.time()-start):.2f} Hz")

        return features
