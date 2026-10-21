import torch
import torch.nn.functional as F
from src.normalization import (
    normalize_images, denormalize_images, loader_normalization, normalization_for_loader,
)


# ---------------------------------------------------------
# Batch / model helpers
# ---------------------------------------------------------

def unpack_batch(batch):

    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError(f"Expected at least 2 batch elements, got {len(batch)}")
        return batch[0], batch[1]

    if isinstance(batch, dict):
        image_keys = ["image", "images", "img", "input", "inputs"]
        label_keys = ["label", "labels", "target", "targets", "class", "class_idx", "class_id", "y"]

        images = next((batch[k] for k in image_keys if k in batch), None)
        labels = next((batch[k] for k in label_keys if k in batch), None)

        if images is None:
            raise KeyError(f"Could not find images in batch. Available keys: {list(batch.keys())}")
        if labels is None:
            raise KeyError(f"Could not find labels in batch. Available keys: {list(batch.keys())}")

        return images, labels

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def get_logits(model_output):
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


# ---------------------------------------------------------
# FGSM
# ---------------------------------------------------------

def fgsm_attack(model, images, labels, epsilon, mean=None, std=None):

    clean_pixels = denormalize_images(images.detach(), mean, std).clamp(0.0, 1.0)

    adv_pixels = clean_pixels.clone().detach()
    adv_pixels.requires_grad_(True)

    model_input = normalize_images(adv_pixels, mean, std)
    logits = get_logits(model(model_input))
    loss = F.cross_entropy(logits, labels)

    grad = torch.autograd.grad(loss, adv_pixels, retain_graph=False, create_graph=False)[0]

    with torch.no_grad():
        adv_pixels = adv_pixels + epsilon * grad.sign()
        delta = torch.clamp(adv_pixels - clean_pixels, min=-epsilon, max=epsilon)
        adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0)

    adv_images = normalize_images(adv_pixels, mean, std)
    return adv_images.detach()


# ---------------------------------------------------------
# PGD
# ---------------------------------------------------------

def pgd_attack(model, images, labels, epsilon, alpha, steps, random_start=True, mean=None, std=None):

    clean_pixels = denormalize_images(images.detach(), mean, std).clamp(0.0, 1.0)

    if random_start:
        adv_pixels = clean_pixels + torch.empty_like(clean_pixels).uniform_(-epsilon, epsilon)
        adv_pixels = adv_pixels.clamp(0.0, 1.0)
    else:
        adv_pixels = clean_pixels.clone()

    for _ in range(steps):
        adv_pixels = adv_pixels.detach()
        adv_pixels.requires_grad_(True)

        model_input = normalize_images(adv_pixels, mean, std)
        logits = get_logits(model(model_input))
        loss = F.cross_entropy(logits, labels)

        grad = torch.autograd.grad(loss, adv_pixels, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            adv_pixels = adv_pixels + alpha * grad.sign()
            delta = torch.clamp(adv_pixels - clean_pixels, min=-epsilon, max=epsilon)
            adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0)

    adv_images = normalize_images(adv_pixels, mean, std)
    return adv_images.detach()


# ---------------------------------------------------------
# Clean evaluation
# ---------------------------------------------------------

@torch.no_grad()
def clean_accuracy(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    for batch in loader:
        images, labels = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        logits = get_logits(model(images))
        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------
# Adversarial evaluation
# ---------------------------------------------------------

def adversarial_accuracy(model, loader, device, attack, **attack_kwargs):
    model.eval()

    if loader_normalization(loader) is not None:
        mean, std = normalization_for_loader(loader, attack_kwargs.get("mean"), attack_kwargs.get("std"))
        attack_kwargs.update(mean=mean, std=std)

    correct = 0
    total = 0

    for batch in loader:
        images, labels = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        # Attack requires gradient wrt input
        with torch.enable_grad():
            adv_images = attack(model=model, images=images, labels=labels, **attack_kwargs)

        # Evaluation itself does not need gradients
        with torch.no_grad():
            logits = get_logits(model(adv_images))
            predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------
# Full robustness evaluation
# ---------------------------------------------------------

def evaluate_adversarial_robustness(model, loader, device, mean=None, std=None, epsilons=None, pgd_steps=10):
    """
    Evaluate clean + FGSM + PGD for multiple epsilons.

    Returns:
        list of dicts
    """
    if epsilons is None:
        epsilons = [1 / 255, 2 / 255, 4 / 255, 8 / 255]

    model.eval()

    print("\n======================================")
    print("Adversarial robustness evaluation")
    print("======================================")

    clean_acc = clean_accuracy(model, loader, device)
    print(f"Clean accuracy: {clean_acc:.4f} ({100 * clean_acc:.2f}%)")

    results = []

    for epsilon in epsilons:
        # Common PGD heuristic
        alpha = max(epsilon / 4, 1 / 255)

        fgsm_acc = adversarial_accuracy(
            model=model, loader=loader, device=device,
            attack=fgsm_attack, epsilon=epsilon, mean=mean, std=std,
        )

        pgd_acc = adversarial_accuracy(
            model=model, loader=loader, device=device,
            attack=pgd_attack, epsilon=epsilon, alpha=alpha,
            steps=pgd_steps, random_start=True, mean=mean, std=std,
        )

        result = {
            "epsilon": epsilon,
            "epsilon_255": epsilon * 255,
            "clean_acc": clean_acc,
            "fgsm_acc": fgsm_acc,
            "pgd_acc": pgd_acc,
        }
        results.append(result)

        print(f"\nepsilon = {epsilon * 255:.0f}/255")
        print(f"FGSM accuracy: {fgsm_acc:.4f} ({100 * fgsm_acc:.2f}%)")
        print(f"PGD-{pgd_steps} accuracy: {pgd_acc:.4f} ({100 * pgd_acc:.2f}%)")

    print("\n======================================")

    return results
