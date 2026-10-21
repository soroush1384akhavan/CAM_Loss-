"""RGB input normalization shared by data, reports and pixel-space attacks."""
import math
import warnings

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BIT_MEAN = (0.5, 0.5, 0.5)
BIT_STD = (0.5, 0.5, 0.5)


def validate_normalization(mean, std):
    try:
        mean, std = tuple(float(x) for x in mean), tuple(float(x) for x in std)
    except (TypeError, ValueError) as error:
        raise ValueError("Normalization needs both mean and std, each with three RGB numbers") from error
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("Normalization mean and std must each contain three RGB numbers")
    if not all(math.isfinite(x) for x in (*mean, *std)) or any(x <= 0 for x in std):
        raise ValueError("Normalization values must be finite and every std must be positive")
    return mean, std


def resolve_normalization(config):
    """Explicit RGB statistics win; otherwise use the selected model's defaults."""
    mean, std = config.get("normalize_mean"), config.get("normalize_std")
    if mean is None and std is None:
        # adapter_clip is currently an alias for the BiT model in model_factory.
        if str(config.get("model_type", "plain")).lower() in {"adapter_bit", "adapter_clip"}:
            mean, std = BIT_MEAN, BIT_STD
        else:
            mean, std = IMAGENET_MEAN, IMAGENET_STD
    mean, std = validate_normalization(mean, std)
    attack_mean, attack_std = config.get("attack_normalize_mean"), config.get("attack_normalize_std")
    if attack_mean is not None or attack_std is not None:
        attack = validate_normalization(attack_mean, attack_std)
        if attack != (mean, std):
            raise ValueError("attack_normalize_mean/std must match normalize_mean/std; remove the attack overrides")
    return mean, std


def configure_normalization(config):
    """Record resolved defaults so the saved config describes the actual input."""
    mean, std = resolve_normalization(config)
    config["normalize_mean"], config["normalize_std"] = list(mean), list(std)
    return mean, std


def loader_normalization(loader):
    """Read the actual transform statistics, including through nested Subsets."""
    dataset = getattr(loader, "dataset", None)
    while dataset is not None:
        if hasattr(dataset, "normalize_mean") and hasattr(dataset, "normalize_std"):
            return validate_normalization(dataset.normalize_mean, dataset.normalize_std)
        dataset = getattr(dataset, "dataset", None)
    return None


def normalization_for_loader(loader, mean=None, std=None):
    actual = loader_normalization(loader)
    if mean is None and std is None:
        return actual if actual is not None else (IMAGENET_MEAN, IMAGENET_STD)
    requested = validate_normalization(mean, std)
    if actual is not None and requested != actual:
        raise ValueError(f"Normalization mismatch: loader uses {actual}, caller requested {requested}")
    return requested


def _stats(images, mean, std):
    mean, std = validate_normalization(mean, std)
    if images.ndim not in (3, 4) or images.shape[-3] != 3 or not images.is_floating_point():
        raise ValueError("Expected floating-point RGB image(s) shaped [3,H,W] or [B,3,H,W]")
    shape = (3, 1, 1) if images.ndim == 3 else (1, 3, 1, 1)
    return images.new_tensor(mean).view(shape), images.new_tensor(std).view(shape)


def normalize_images(images, mean, std):
    """Pixels in [0,1] -> model input. Both None explicitly means pixel input."""
    if mean is None and std is None:
        return images
    mean_t, std_t = _stats(images, mean, std)
    return (images - mean_t) / std_t


def denormalize_images(images, mean, std):
    """Inverse input normalization, without clipping or changing contrast."""
    if mean is None and std is None:
        return images
    mean_t, std_t = _stats(images, mean, std)
    return images * std_t + mean_t


def validate_checkpoint_normalization(checkpoint, mean, std):
    requested = validate_normalization(mean, std)
    metadata = checkpoint.get("input_normalization") if isinstance(checkpoint, dict) else None
    if metadata is None:
        warnings.warn(
            "Checkpoint has no input_normalization metadata. Verify its training transform: "
            "older src/data.py hardcoded ImageNet mean/std even when config.yaml said 0.5. "
            "Use explicit ImageNet normalize_mean/std to reproduce those runs.",
            UserWarning, stacklevel=2,
        )
    elif validate_normalization(metadata["mean"], metadata["std"]) != requested:
        raise ValueError(f"Checkpoint normalization {metadata} does not match evaluation normalization {requested}")
