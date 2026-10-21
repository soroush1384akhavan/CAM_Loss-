import random
from pathlib import Path
import csv
from typing import Dict, Optional, Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms, datasets
from torchvision.transforms import functional as TF

from src.normalization import IMAGENET_MEAN, IMAGENET_STD, validate_normalization


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _normalize_mapping(mapping: Optional[dict]) -> Dict[str, Path]:
    if mapping is None:
        return {}
    return {str(k): Path(v) for k, v in mapping.items()}


def load_valid_mask_stems(manifest_path: Path) -> Optional[set]:

    if not manifest_path or not manifest_path.exists():
        return None

    valid = set()

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            has_mask = str(row.get("has_mask", "0")).strip()
            status = str(row.get("status", "")).strip().lower()
            mask_path = row.get("mask_path", "")

            if has_mask == "1" and status == "ok":
                if mask_path:
                    valid.add(Path(mask_path).stem)
                else:
                    image_path = row.get("image_path", "")
                    if image_path:
                        valid.add(Path(image_path).stem)

    return valid


class ImageFolderWithClassMasks(Dataset):
    """
    Generic multi-class ImageFolder dataset with optional pseudo-mask per class.

    Expected image layout:
        data_dir/
          class_a/*.jpg
          class_b/*.jpg
          ...

    Mask paths are given through config:
        mask_dirs:
          class_a: path/to/class_a/masks
          class_b: path/to/class_b/masks

    Optional manifests:
        mask_manifests:
          class_a: path/to/class_a/manifest.csv
          class_b: path/to/class_b/manifest.csv

    Mask filename must match the image stem:
        image: dog_000001.jpg
        mask:  dog_000001.png   
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int,
        mask_dirs: Optional[dict] = None,
        mask_manifests: Optional[dict] = None,
        normalize_mean=None,
        normalize_std=None,
        augment_crop_scale=(0.7, 1.0),
        augment_hflip: bool = True,
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.data_dir}")

        self.image_size = image_size
        if normalize_mean is None and normalize_std is None:
            normalize_mean, normalize_std = IMAGENET_MEAN, IMAGENET_STD
        self.normalize_mean, self.normalize_std = validate_normalization(normalize_mean, normalize_std)
        self.mask_dirs = _normalize_mapping(mask_dirs)
        self.mask_manifests = _normalize_mapping(mask_manifests)
        self.augment_crop_scale = tuple(float(v) for v in augment_crop_scale)
        self.augment_hflip = bool(augment_hflip)

        # Use ImageFolder only to build samples/classes/class_to_idx reliably.
        base_dataset = datasets.ImageFolder(
            root=str(self.data_dir),
            transform=None,
            is_valid_file=is_valid_image,
        )

        self.samples: List[Tuple[str, int]] = base_dataset.samples
        self.classes = base_dataset.classes
        self.class_to_idx = base_dataset.class_to_idx

        self.valid_mask_stems_by_class = {}
        for class_name in self.classes:
            manifest_path = self.mask_manifests.get(class_name)
            if manifest_path is not None:
                self.valid_mask_stems_by_class[class_name] = load_valid_mask_stems(manifest_path)
            else:
                self.valid_mask_stems_by_class[class_name] = None

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.normalize_mean,
                std=self.normalize_std,
            ),
        ])
        self.image_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.normalize_mean,
                std=self.normalize_std,
            ),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

        print("Classes:", self.classes)
        print("Total samples:", len(self.samples))
        for class_name in self.classes:
            mask_dir = self.mask_dirs.get(class_name)
            manifest = self.mask_manifests.get(class_name)
            print(f"Class '{class_name}': mask_dir={mask_dir}, manifest={manifest}")

    def __len__(self):
        return len(self.samples)

    def _load_mask_image(self, image_path: Path, label: int) -> Optional[Image.Image]:
        class_name = self.classes[label]
        mask_dir = self.mask_dirs.get(class_name)

        if mask_dir is None:
            return None

        mask_path = mask_dir / f"{image_path.stem}.png"
        valid_stems = self.valid_mask_stems_by_class.get(class_name)

        # If manifest exists, require the stem to be valid in manifest.
        if valid_stems is not None and image_path.stem not in valid_stems:
            return None

        if not mask_path.exists():
            return None

        return Image.open(mask_path).convert("L")

    def _mask_to_tensor(self, mask: Optional[Image.Image]):
        mask_tensor = torch.zeros(1, self.image_size, self.image_size)
        has_mask = torch.tensor(0.0, dtype=torch.float32)

        if mask is None:
            return mask_tensor, has_mask

        mask_tensor = (TF.to_tensor(mask) > 0.5).float()

        # A random crop can cut the object out entirely; such a sample then
        # carries no mask supervision rather than an all-background one.
        if mask_tensor.sum() > 0:
            has_mask = torch.tensor(1.0, dtype=torch.float32)

        return mask_tensor, has_mask

    def _load_mask(self, image_path: Path, label: int):
        mask = self._load_mask_image(image_path, label)
        if mask is not None:
            mask = TF.resize(
                mask,
                [self.image_size, self.image_size],
                interpolation=transforms.InterpolationMode.NEAREST,
            )
        return self._mask_to_tensor(mask)

    def _load_augmented(self, image: Image.Image, image_path: Path, label: int):
        """Same random crop and flip on the image and its mask."""
        size = [self.image_size, self.image_size]
        mask = self._load_mask_image(image_path, label)
        if mask is not None and mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)

        top, left, height, width = transforms.RandomResizedCrop.get_params(
            image, scale=self.augment_crop_scale, ratio=(3 / 4, 4 / 3)
        )
        image = TF.resized_crop(image, top, left, height, width, size)
        if mask is not None:
            mask = TF.resized_crop(
                mask, top, left, height, width, size,
                interpolation=transforms.InterpolationMode.NEAREST,
            )

        if self.augment_hflip and torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)

        mask_tensor, has_mask = self._mask_to_tensor(mask)
        return self.image_to_tensor(image), mask_tensor, has_mask

    def load(self, idx, augment: bool = False):
        image_path_str, label = self.samples[idx]
        image_path = Path(image_path_str)

        image = Image.open(image_path).convert("RGB")

        if augment:
            image_tensor, mask_tensor, has_mask = self._load_augmented(image, image_path, label)
        else:
            image_tensor = self.image_transform(image)
            mask_tensor, has_mask = self._load_mask(image_path, label)

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "mask": mask_tensor,
            "has_mask": has_mask,
            "image_path": str(image_path),
        }

    def __getitem__(self, idx):
        return self.load(idx, augment=False)


class AugmentedSubset(Subset):
    """Training split: same indices as a Subset, loaded with joint augmentation."""

    def __getitem__(self, idx):
        return self.dataset.load(self.indices[idx], augment=True)


def _subsample_train_indices(
    base_dataset: ImageFolderWithClassMasks,
    indices: List[int],
    shots_per_class: Optional[int] = None,
    fraction: Optional[float] = None,
    seed: int = 42,
) -> List[int]:
    
    if shots_per_class is None and fraction is None:
        return indices

    if shots_per_class is not None and fraction is not None:
        raise ValueError(
            "Specify either shots_per_class or fraction, not both."
        )

    rng = random.Random(seed)

    by_class: Dict[int, List[int]] = {}
    for idx in indices:
        _, label = base_dataset.samples[idx]
        by_class.setdefault(label, []).append(idx)

    selected: List[int] = []
    for label, idxs in by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)

        if shots_per_class is not None:
            k = min(shots_per_class, len(idxs))
        else:
            k = max(1, round(len(idxs) * fraction))
            k = min(k, len(idxs))

        chosen = idxs[:k]
        selected.extend(chosen)

        class_name = base_dataset.classes[label]
        print(f"  [subsample] class='{class_name}': {len(chosen)}/{len(idxs)} samples kept for training")

    rng.shuffle(selected)
    return selected


def get_dataloaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    val_ratio: float,
    seed: int,
    num_workers: int,
    mask_dirs: Optional[dict] = None,
    mask_manifests: Optional[dict] = None,
    test_ratio: float = 0.0,
    train_shots_per_class: Optional[int] = None,
    train_fraction: Optional[float] = None,
    normalize_mean=None,
    normalize_std=None,
    train_augment: bool = False,
    augment_crop_scale=(0.7, 1.0),
    augment_hflip: bool = True,
):

    dataset = ImageFolderWithClassMasks(
        data_dir=data_dir,
        image_size=image_size,
        mask_dirs=mask_dirs,
        mask_manifests=mask_manifests,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        augment_crop_scale=augment_crop_scale,
        augment_hflip=augment_hflip,
    )

    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size - test_size

    if train_size <= 0:
        raise ValueError(
            f"val_ratio + test_ratio too large for dataset size {total_size} "
            f"(train_size would be {train_size})."
        )

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    print(
        f"Split sizes | train={train_size} val={val_size} test={test_size} "
        f"(total={total_size})"
    )

    if train_shots_per_class is not None or train_fraction is not None:
        print("Subsampling training set:")
        selected_indices = _subsample_train_indices(
            base_dataset=dataset,
            indices=train_dataset.indices,
            shots_per_class=train_shots_per_class,
            fraction=train_fraction,
            seed=seed,
        )
        train_dataset = Subset(dataset, selected_indices)
        print(f"Training set size after subsampling: {len(train_dataset)}")

    if train_augment:
        train_dataset = AugmentedSubset(dataset, list(train_dataset.indices))
        print(
            f"Train augmentation: RandomResizedCrop(scale={tuple(augment_crop_scale)}) "
            f"+ hflip={augment_hflip}, applied jointly to image and mask"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, dataset.classes
