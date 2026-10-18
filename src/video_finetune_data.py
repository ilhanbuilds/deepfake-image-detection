"""Verified historical acceptance roster; freshly decoded pixels, never features."""

import torch

from src.video_pipeline import VideoDataset


class CropDriftError(RuntimeError):
    pass


class VerifiedImageDataset(VideoDataset):
    def __init__(self, root, records, settings, cache):
        super().__init__(root, records, settings)
        self.expected = {}
        # Cache was fully audited first. Retain crop metadata ONLY, no tensors.
        for record in records:
            saved = torch.load(
                cache.path(record), map_location="cpu", weights_only=True
            )
            self.expected[record["path"]] = saved["report"]

    def verify(self, record, frames, report):
        expected = self.expected[record["path"]]
        if frames is None or any(
            report.get(key) != expected.get(key)
            for key in (
                "status",
                "valid_indices",
                "selected_indices",
                "boxes",
                "failed_frames",
            )
        ):
            raise CropDriftError(
                f"Yüz crop/kabul durumu değişti: {record['path']}. "
                "Video sessizce atılmadı; sınıf ağırlığı/kapsam korunmak için run durduruldu."
            )

    def __getitem__(self, index):
        item = super().__getitem__(index)
        self.verify(item["record"], item["values"], item["report"])
        return item
