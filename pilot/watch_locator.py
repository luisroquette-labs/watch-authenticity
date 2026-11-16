"""Find the watch before judging photo quality or visible reference details."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

MODEL_ID = "google/owlv2-base-patch16-ensemble"
MODEL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
CACHE_DIR = Path.home() / ".cache/watch-authenticity/detector-cache"
QUERIES = ("wristwatch", "watch face")


@dataclass(frozen=True)
class WatchRegion:
    watch: Image.Image
    face: Image.Image | None
    watch_score: float
    face_score: float | None


class LocalWatchLocator:
    def __init__(self) -> None:
        self.processor = Owlv2Processor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=CACHE_DIR, local_files_only=True
        )
        self.model = Owlv2ForObjectDetection.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=CACHE_DIR, local_files_only=True
        ).eval()

    def locate(self, image: Image.Image) -> WatchRegion | None:
        inputs = self.processor(text=[list(QUERIES)], images=image, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            output, threshold=0.2, target_sizes=[image.size[::-1]]
        )[0]
        candidates = [
            (float(score), int(label), tuple(float(value) for value in box))
            for score, label, box in zip(
                result["scores"], result["labels"], result["boxes"]
            )
        ]
        return choose_region(image, candidates)


def choose_region(
    image: Image.Image, candidates: list[tuple[float, int, tuple[float, ...]]]
) -> WatchRegion | None:
    def valid(box: tuple[float, ...]) -> bool:
        return (
            len(box) == 4
            and min(image.width, box[2]) - max(0, box[0]) >= 40
            and min(image.height, box[3]) - max(0, box[1]) >= 40
        )

    def crop(box: tuple[float, ...]) -> Image.Image:
        return image.crop((max(0, box[0]), max(0, box[1]), min(image.width, box[2]), min(image.height, box[3])))

    watches = [(score, box) for score, label, box in candidates if label == 0 and score >= 0.35 and valid(box)]
    faces = [(score, box) for score, label, box in candidates if label == 1 and score >= 0.2 and valid(box)]
    if watches:
        watch_score, watch_box = max(watches, key=lambda item: item[0])
        faces = [
            (score, box) for score, box in faces
            if watch_box[0] <= (box[0] + box[2]) / 2 <= watch_box[2]
            and watch_box[1] <= (box[1] + box[3]) / 2 <= watch_box[3]
            and box[2] - box[0] <= watch_box[2] - watch_box[0]
            and box[3] - box[1] <= watch_box[3] - watch_box[1]
        ]
    elif faces and max(faces, key=lambda item: item[0])[0] >= 0.45:
        watch_score, watch_box = max(faces, key=lambda item: item[0])
    else:
        return None

    face_score, face_box = max(faces, key=lambda item: item[0]) if faces else (None, None)
    return WatchRegion(
        watch=crop(watch_box),
        face=crop(face_box) if face_box else None,
        watch_score=watch_score,
        face_score=face_score,
    )
