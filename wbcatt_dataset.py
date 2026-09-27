import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


NUM_CLASSES = 6

CLASS_NAMES: tuple[str, ...] = (
    "background",
    "cytoplasm",
    "nucleus",
    "platelets",
    "RBC",
    "vacuoles",
)

CLASS_COLORS = np.asarray(
    [
        (0, 0, 0),
        (255, 165, 0),
        (0, 200, 255),
        (0, 255, 120),
        (255, 40, 110),
        (170, 80, 255),
    ],
    dtype=np.uint8,
)

SPLIT_FILES: Mapping[str, str] = {
    "train": "pbc_attr_v1_ccrop_train.csv",
    "val": "pbc_attr_v1_ccrop_val.csv",
    "test": "pbc_attr_v1_ccrop_test.csv",
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def mask_name_from_image(image_name: str) -> str:
    image_path = Path(image_name)
    return f"{image_path.stem}_mask.png"


def colorize_mask(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}.")
    if mask.size and (mask.min() < 0 or mask.max() >= NUM_CLASSES):
        raise ValueError(
            f"Mask values must be in [0, {NUM_CLASSES - 1}], "
            f"got min={mask.min()} and max={mask.max()}."
        )
    return CLASS_COLORS[mask.astype(np.int64)]


class WBCAttDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        csv_file: str | Path,
        image_size: tuple[int, int] = (360, 360),
        augment: bool = False,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        validate_files: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.csv_file = Path(csv_file)
        self.image_size = tuple(int(v) for v in image_size)
        self.augment = augment

        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must contain two positive integers.")
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")
        if not self.csv_file.is_file():
            raise FileNotFoundError(f"Split CSV does not exist: {self.csv_file}")

        self.frame = pd.read_csv(self.csv_file)
        required_columns = {"img_name", "path", "label", "split"}
        missing_columns = required_columns.difference(self.frame.columns)
        if missing_columns:
            raise ValueError(
                f"{self.csv_file.name} is missing columns: {sorted(missing_columns)}"
            )

        self.frame = self.frame.reset_index(drop=True)
        self.image_names = [Path(str(path)).name for path in self.frame["path"]]
        duplicated = pd.Series(self.image_names).duplicated()
        if duplicated.any():
            examples = pd.Series(self.image_names)[duplicated].head(5).tolist()
            raise ValueError(f"Duplicate image names in {self.csv_file.name}: {examples}")

        if (mean is None) != (std is None):
            raise ValueError("mean and std must either both be provided or both be None.")
        if mean is None:
            self.mean = None
            self.std = None
        else:
            if len(mean) != 3 or len(std) != 3:
                raise ValueError("mean and std must each contain three RGB values.")
            self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
            self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
            if torch.any(self.std <= 0):
                raise ValueError("All std values must be positive.")

        if validate_files:
            missing = []
            for image_name in self.image_names:
                image_path = self.data_dir / image_name
                mask_path = self.data_dir / mask_name_from_image(image_name)
                if not image_path.is_file() or not mask_path.is_file():
                    missing.append((str(image_path), str(mask_path)))
                    if len(missing) == 5:
                        break
            if missing:
                raise FileNotFoundError(f"Missing image/mask pairs, examples: {missing}")

    def __len__(self) -> int:
        return len(self.frame)

    def _resize_pair(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        height, width = self.image_size
        output_size = (width, height)
        if image.size != output_size:
            image = image.resize(output_size, resample=Image.Resampling.BILINEAR)
        if mask.size != output_size:
            mask = mask.resize(output_size, resample=Image.Resampling.NEAREST)
        return image, mask

    def _augment_pair(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        rotation = random.randrange(4)
        if rotation:
            angle = 90 * rotation
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        image_name = self.image_names[index]
        image_path = self.data_dir / image_name
        mask_path = self.data_dir / mask_name_from_image(image_name)

        with Image.open(image_path) as loaded_image:
            image = loaded_image.convert("RGB")
        with Image.open(mask_path) as loaded_mask:
            mask = loaded_mask.convert("L")

        image, mask = self._resize_pair(image, mask)
        if self.augment:
            image, mask = self._augment_pair(image, mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.int64)

        mask_min = int(mask_array.min())
        mask_max = int(mask_array.max())
        if mask_min < 0 or mask_max >= NUM_CLASSES:
            raise ValueError(
                f"Invalid mask values in {mask_path.name}: min={mask_min}, max={mask_max}."
            )

        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(mask_array.copy()).long()
        if self.mean is not None and self.std is not None:
            image_tensor = (image_tensor - self.mean) / self.std

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_name": image_name,
            "cell_type": str(row["label"]),
            "split": str(row["split"]),
        }


def create_datasets(
    data_dir: str | Path,
    csv_dir: str | Path,
    image_size: tuple[int, int] = (360, 360),
    train_augment: bool = True,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    validate_files: bool = False,
) -> dict[str, WBCAttDataset]:
    csv_dir = Path(csv_dir)
    return {
        split: WBCAttDataset(
            data_dir=data_dir,
            csv_file=csv_dir / filename,
            image_size=image_size,
            augment=train_augment and split == "train",
            mean=mean,
            std=std,
            validate_files=validate_files,
        )
        for split, filename in SPLIT_FILES.items()
    }


def create_dataloaders(
    datasets: Mapping[str, Dataset],
    batch_size: int = 4,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    loaders: dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
        )
    return loaders


def validate_official_splits(
    data_dir: str | Path,
    csv_dir: str | Path,
) -> dict[str, object]:
    data_dir = Path(data_dir)
    csv_dir = Path(csv_dir)
    split_names: dict[str, set[str]] = {}
    split_rows: dict[str, int] = {}
    declared_splits: dict[str, list[str]] = {}
    duplicate_counts: dict[str, int] = {}

    for split, filename in SPLIT_FILES.items():
        csv_path = csv_dir / filename
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")
        frame = pd.read_csv(csv_path)
        if "path" not in frame.columns:
            raise ValueError(f"Missing 'path' column in {csv_path.name}")
        names = [Path(str(path)).name for path in frame["path"]]
        split_rows[split] = len(names)
        duplicate_counts[split] = len(names) - len(set(names))
        split_names[split] = set(names)
        declared_splits[split] = sorted(frame["split"].astype(str).unique().tolist())

    overlaps = {
        "train_val": len(split_names["train"] & split_names["val"]),
        "train_test": len(split_names["train"] & split_names["test"]),
        "val_test": len(split_names["val"] & split_names["test"]),
    }
    csv_names = set().union(*split_names.values())
    disk_names = {path.name for path in data_dir.glob("*.jpg")}
    missing_images = sorted(csv_names - disk_names)
    missing_masks = sorted(
        image_name
        for image_name in csv_names
        if not (data_dir / mask_name_from_image(image_name)).is_file()
    )

    return {
        "split_rows": split_rows,
        "declared_splits": declared_splits,
        "duplicate_counts": duplicate_counts,
        "overlaps": overlaps,
        "csv_unique_images": len(csv_names),
        "disk_unique_images": len(disk_names),
        "disk_images_not_in_csv": sorted(disk_names - csv_names),
        "csv_images_not_on_disk": missing_images,
        "missing_masks": missing_masks,
        "is_valid": (
            not any(duplicate_counts.values())
            and not any(overlaps.values())
            and not missing_images
            and not missing_masks
            and disk_names == csv_names
        ),
    }


def pixel_class_counts(
    mask_paths: Iterable[str | Path],
    max_images: int | None = None,
) -> np.ndarray:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for index, mask_path in enumerate(mask_paths):
        if max_images is not None and index >= max_images:
            break
        with Image.open(mask_path) as loaded_mask:
            mask = np.asarray(loaded_mask.convert("L"), dtype=np.uint8)
        values = np.unique(mask)
        if values.size and (values.min() < 0 or values.max() >= NUM_CLASSES):
            raise ValueError(f"Invalid labels {values.tolist()} in {mask_path}")
        counts += np.bincount(mask.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    return counts


if __name__ == "__main__":
    local_root = Path(__file__).resolve().parent
    report = validate_official_splits(
        data_dir=local_root / "pbcseg_final_v1" / "pbcseg_final_v1",
        csv_dir=local_root / "dataset_txt",
    )
    print(report)
