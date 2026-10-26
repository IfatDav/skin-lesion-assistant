"""Image dataset, transforms and loaders built on data/splits/image_index.csv."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from src import config as C

_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def train_transform(size: int = C.IMG_SIZE):
    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(180),
        # Hue stays nearly fixed: lesion colour is diagnostic.
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        T.ToDtype(torch.float32, scale=True), _NORM])


def eval_transform(size: int = C.IMG_SIZE):
    return T.Compose([T.ToImage(), T.Resize(size, antialias=True), T.CenterCrop(size),
                      T.ToDtype(torch.float32, scale=True), _NORM])


def load_index(splits=None, skip_duplicates: bool = True) -> pd.DataFrame:
    df = pd.read_csv(C.INDEX_CSV, dtype=str).fillna("")
    if splits is not None:
        df = df[df["split"].isin([splits] if isinstance(splits, str) else splits)]
    if skip_duplicates:
        df = df[df["duplicate_of"] == ""]
    return df.reset_index(drop=True)


def resolve_path(row) -> Path:
    if row["source"] == "pad":
        cached = C.PAD_RESIZED_DIR / (Path(row["image_path"]).stem + ".jpg")
        if cached.exists():
            return cached
    return C.DATA_ROOT / row["image_path"]


class LesionImages(Dataset):
    """Returns (image tensor, class index or -1 when the image has no 6-class label)."""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = [resolve_path(r) for _, r in df.iterrows()]
        self.labels = [C.CLASSES.index(l) if l in C.CLASSES else -1 for l in df["label6"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def make_loader(df: pd.DataFrame, train: bool, batch_size: int = 32, workers: int = 8) -> DataLoader:
    ds = LesionImages(df, train_transform() if train else eval_transform())
    return DataLoader(ds, batch_size=batch_size if train else 64, shuffle=train, drop_last=train,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
