from typing import List
import time
import torch
import numpy as np
from .OSNet import osnet_x0_5  # OSNet-0.5 variant
from .utils import load_pretrained_weights
from .image_handler import normalize, resize, ndarray_to_tensor

class OsNetEncoder:

    LOSS = 'softmax'

    def __init__(
        self,
        input_width: int,
        input_height: int,
        weight_filepath: str,
        batch_size: int = 32,
        num_classes: int = 751,  # Market1501 classes
        patch_height: int = 256,
        patch_width: int = 128,
        norm_mean: List[float] = [0.485, 0.456, 0.406],
        norm_std: List[float] = [0.229, 0.224, 0.225],
        GPU: bool = True
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

        # Load OSNet-0.5 model
        self._model = osnet_x0_5(
            num_classes=self.num_classes,
            loss=OsNetEncoder.LOSS
        )

        # Load pretrained ReID weights
        load_pretrained_weights(self._model, self._weight_filepath)

        self._model.eval()
        if self.GPU:
            self._model = self._model.cuda()

    def load_image(self, patch: np.ndarray) -> torch.Tensor:
        """Resize, normalize, and convert a single patch to a 4D batch tensor [1, C, H, W]."""
        resized = resize(patch, self.patch_size)
        tensor = ndarray_to_tensor(resized)
        tensor = normalize(tensor, self.norm_mean, self.norm_std, inplace=False)

        # Ensure batch dimension exists: [1, C, H, W]
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.to(torch.device("cuda" if self.GPU else "cpu"))

    def get_features(self, image_patches: List[np.ndarray]) -> List[np.ndarray]:
        """Extract L2-normalized features for a list of image patches."""
        features = []

        for patch in image_patches:
            if patch is None:
                features.append(None)
                continue

            start = time.time()
            patch_tensor = self.load_image(patch)

            with torch.no_grad():
                feat = self._model(patch_tensor)

            # flatten and L2-normalize
            feat = feat.cpu().numpy().reshape(-1).astype(np.float32)
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat /= norm

            features.append(feat)
            print(f"[PERFORMANCE] Features extracted: {1/(time.time()-start):.2f} Hz")

        return features
