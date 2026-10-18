"""V3 için yalnızca train'de fotoğraf kalite augmentation'ı; CLEAN eval değişmez."""

import io
import math
from dataclasses import dataclass

import torch
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF

from src.dataset import IMAGE_SIZE, image_transform
from src.presplit_dataset import PresplitFaceDataset


@dataclass(frozen=True)
class AugmentationSettings:
    jpeg_probability: float = 0.50
    jpeg_quality: tuple[int, int] = (50, 95)
    blur_probability: float = 0.15
    blur_sigma: tuple[float, float] = (0.1, 1.5)
    noise_probability: float = 0.15
    noise_std: tuple[float, float] = (0.0, 0.02)
    resize_probability: float = 0.30
    resize_size: tuple[int, int] = (112, 224)
    flip_probability: float = 0.50

    def validate(self):
        for probability in (
            self.jpeg_probability,
            self.blur_probability,
            self.noise_probability,
            self.resize_probability,
            self.flip_probability,
        ):
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Augmentation olasılığı [0,1] içinde olmalı.")
        for low, high in (self.jpeg_quality, self.resize_size):
            if not isinstance(low, int) or not isinstance(high, int) or low > high:
                raise ValueError("JPEG/resize aralıkları sıralı tam sayı olmalı.")
        if not 1 <= self.jpeg_quality[0] <= self.jpeg_quality[1] <= 100:
            raise ValueError("JPEG quality [1,100] içinde olmalı.")
        if not 1 <= self.resize_size[0] <= self.resize_size[1] <= IMAGE_SIZE:
            raise ValueError("Downscale aralığı [1,224] içinde olmalı.")
        for low, high in (self.blur_sigma, self.noise_std):
            if not all(math.isfinite(v) for v in (low, high)) or not 0 <= low <= high:
                raise ValueError("Sigma/std aralığı sonlu, sıralı ve negatif olmamalı.")
        if self.blur_sigma[0] <= 0:
            raise ValueError("Blur sigma pozitif olmalı.")


class RobustTrainTransform:
    """Torch RNG kullanır; workers=0 ile checkpoint RNG resume'a dahildir.

    JPEG/blur/resize Pillow üzerinde, noise RGB [0,1] tensor üzerinde.
    Pillow GaussianBlur radius standart sapmadır; genişletilmiş box filter
    dizisiyle Gaussian'ı yaklaşıklar ve CPU/Windows workers=0 için hafiftir.
    Her gate bağımsızdır; bozulmalar aynı fotoğrafta birlikte uygulanabilir.
    """

    def __init__(self, settings=None):
        self.settings = settings or AugmentationSettings()
        self.settings.validate()

    @staticmethod
    def _apply(probability):
        return torch.rand(()).item() < probability

    @staticmethod
    def _uniform(bounds):
        low, high = bounds
        return low + (high - low) * torch.rand(()).item()

    @staticmethod
    def _integer(bounds):
        return int(torch.randint(bounds[0], bounds[1] + 1, ()).item())

    def __call__(self, image):
        settings = self.settings
        image = TF.resize(image, [IMAGE_SIZE, IMAGE_SIZE], antialias=True)
        if self._apply(settings.flip_probability):
            image = TF.hflip(image)
        if self._apply(settings.jpeg_probability):
            quality = self._integer(settings.jpeg_quality)
            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG", quality=quality, subsampling=2)
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    image = decoded.convert("RGB")
        if self._apply(settings.blur_probability):
            sigma = self._uniform(settings.blur_sigma)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        if self._apply(settings.resize_probability):
            size = self._integer(settings.resize_size)
            image = TF.resize(image, [size, size], antialias=True)
            image = TF.resize(image, [IMAGE_SIZE, IMAGE_SIZE], antialias=True)
        tensor = TF.to_tensor(image)
        if self._apply(settings.noise_probability):
            std = self._uniform(settings.noise_std)
            tensor = (tensor + torch.randn_like(tensor) * std).clamp(0, 1)
        return TF.normalize(
            tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )


def image_transform_v3(training=False, settings=None):
    return RobustTrainTransform(settings) if training else image_transform(False)


class RobustPresplitDataset(PresplitFaceDataset):
    """Train augmentation; valid CLEAN. Training tarafı test splitini reddeder."""

    def __init__(self, root, split, settings=None):
        super().__init__(root, split)
        self.transform = image_transform_v3(split == "train", settings)
