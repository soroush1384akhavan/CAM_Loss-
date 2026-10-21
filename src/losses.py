import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# def build_all_cams(features: torch.Tensor, classifier_weight: torch.Tensor) -> torch.Tensor:
#     """
#         cams: [B, num_classes, H, W]
#     """

#     cams = torch.einsum("kc,bchw->bkhw", classifier_weight, features)
#     cams = F.relu(cams)

#     return cams
class CAMContainmentLoss(nn.Module):

    def __init__(
        self,
        outside_weight=1.0,
        inside_weight=1.0,
        eps=1e-6,
    ):
        super().__init__()

        self.outside_weight = outside_weight
        self.inside_weight = inside_weight  # unused; kept for call-site compatibility
        self.eps = eps

    def forward(
        self,
        cams: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        has_mask: torch.Tensor,
    ) -> torch.Tensor:

        has_mask = has_mask.float()

        if has_mask.sum() == 0:
            return cams.sum() * 0.0

        b, k, hc, wc = cams.shape
        batch_idx = torch.arange(b, device=cams.device)

        # cams:       [B, K, Hc, Wc]
        # target_cam: [B, Hc, Wc]
        target_cam = cams[batch_idx, labels]

        # target_cam: [B, 1, Hc, Wc]
        target_cam = target_cam.unsqueeze(1)

        # Mask resizing
        if mask.shape[-2:] != target_cam.shape[-2:]:
            mask = F.interpolate(mask.float(), size=target_cam.shape[-2:], mode="area")

        inside = mask
        outside = 1.0 - mask

        # Only positive activation counts as "energy": negative logit
        # contribution isn't evidence for the class, so it shouldn't be
        # able to dilute or inflate the containment ratio either direction.
        positive = F.relu(target_cam)

        energy_outside = (positive * outside).sum(dim=(1, 2, 3))
        energy_total = positive.sum(dim=(1, 2, 3)) + self.eps

        containment = energy_outside / energy_total

        loss = self.outside_weight * containment

        # for those which have mask we have to calculate CAMLoss
        loss = loss * has_mask

        # return mean
        return loss.sum() / (has_mask.sum() + self.eps)


class CAMSquaredContainmentLoss(nn.Module):
    def __init__(
        self,
        outside_weight=1.0,
        inside_weight=1.0,
        eps=1e-6,
    ):
        super().__init__()

        self.outside_weight = outside_weight
        self.inside_weight = inside_weight  # unused; kept for call-site compatibility
        self.eps = eps

    def forward(
        self,
        cams: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        has_mask: torch.Tensor,
    ) -> torch.Tensor:
        has_mask = has_mask.float()

        if has_mask.sum() == 0:
            return cams.sum() * 0.0

        batch_size = cams.shape[0]
        batch_idx = torch.arange(batch_size, device=cams.device)
        target_cam = cams[batch_idx, labels].unsqueeze(1)

        if mask.shape[-2:] != target_cam.shape[-2:]:
            mask = F.interpolate(mask.float(), size=target_cam.shape[-2:], mode="area")

        outside = 1.0 - mask.float()
        energy = target_cam.square()
        energy_outside = (energy * outside).sum(dim=(1, 2, 3))
        energy_total = energy.sum(dim=(1, 2, 3)) + self.eps

        containment = energy_outside / energy_total
        loss = self.outside_weight * containment * has_mask

        return loss.sum() / (has_mask.sum() + self.eps)


class CAMAllClassSquaredContainmentLoss(nn.Module):
    def __init__(
        self,
        outside_weight=1.0,
        eps=1e-6,
    ):
        super().__init__()
        self.outside_weight = outside_weight
        self.eps = eps

    def forward(
        self,
        cams: torch.Tensor,      # [B, K, H, W]
        mask: torch.Tensor,      # [B, 1, Hm, Wm]
        labels: torch.Tensor,    # kept for API compatibility
        has_mask: torch.Tensor,  # [B]
    ) -> torch.Tensor:

        has_mask = has_mask.float()

        if has_mask.sum() == 0:
            return cams.sum() * 0.0

        B, K, H, W = cams.shape

        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(
                mask.float(),
                size=(H, W),
                mode="area"
            )

        # [B, 1, H, W]
        outside = 1.0 - mask.float()

        # [B, K, H, W]
        energy = cams.square()

        # broadcast outside over K classes
        # -> [B, K]
        energy_outside = (
            energy * outside
        ).sum(dim=(2, 3))

        # [B, K]
        energy_total = (
            energy.sum(dim=(2, 3)) + self.eps
        )

        # containment for every class
        # [B, K]
        containment = energy_outside / energy_total

        # Average over classes
        # [B]
        per_sample_loss = containment.mean(dim=1)

        per_sample_loss = (
            self.outside_weight
            * per_sample_loss
            * has_mask
        )

        return (
            per_sample_loss.sum()
            / (has_mask.sum() + self.eps)
        )

class CAMContainmentPointingLoss(nn.Module):
    def __init__(self, outside_weight=1.0, inside_weight=1.0, margin=0.0, eps=1e-6):
        super().__init__()
        self.outside_weight = outside_weight
        self.inside_weight = inside_weight 
        self.margin = margin
        self.eps = eps

    def forward(self, cams, mask, labels, has_mask):
        has_mask = has_mask.float()
        if has_mask.sum() == 0:
            return cams.sum() * 0.0

        b, k, hc, wc = cams.shape
        batch_idx = torch.arange(b, device=cams.device)
        target_cam = cams[batch_idx, labels].unsqueeze(1)

        if mask.shape[-2:] != target_cam.shape[-2:]:
            mask = F.interpolate(mask.float(), size=target_cam.shape[-2:], mode="area")

        inside = mask
        outside = 1.0 - mask

        energy_outside = (target_cam ** 2 * outside).sum(dim=(1, 2, 3))
        energy_total = target_cam.abs().sum(dim=(1, 2, 3)) + self.eps
        containment = energy_outside / energy_total

        # NEG_INF = torch.finfo(target_cam.dtype).min
        # cam_flat = target_cam.reshape(b, -1)
        # inside_flat = inside.reshape(b, -1)
        # outside_flat = outside.reshape(b, -1)

        # masked_inside = cam_flat.masked_fill(inside_flat == 0, NEG_INF)
        # masked_outside = cam_flat.masked_fill(outside_flat == 0, NEG_INF)

        # max_inside = masked_inside.max(dim=1).values
        # max_outside = masked_outside.max(dim=1).values

        # pointing_penalty = F.relu(max_outside - max_inside + self.margin)

        loss = (
            self.outside_weight * containment
            # + self.inside_weight * pointing_penalty
        )
        loss = loss * has_mask

        return loss.sum() / (has_mask.sum() + self.eps)
class CAMSoftmaxContainmentLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 1.0,
        outside_weight=1.0,
        inside_weight=1.0,
        eps=1e-6,
    ):
        super().__init__()

        temperature = float(temperature)
        if not temperature > 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.temperature = temperature
        self.outside_weight = outside_weight
        self.inside_weight = inside_weight  # unused; kept for call-site compatibility
        self.eps = eps

    def forward(
        self,
        cams: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        has_mask: torch.Tensor,
    ) -> torch.Tensor:

        has_mask = has_mask.float()

        if has_mask.sum() == 0:
            return cams.sum() * 0.0

        b, k, hc, wc = cams.shape
        batch_idx = torch.arange(b, device=cams.device)

        # cams:       [B, K, Hc, Wc]
        # target_cam: [B, Hc, Wc]
        target_cam = cams[batch_idx, labels].float()

        if mask.shape[-2:] != target_cam.shape[-2:]:
            mask = F.interpolate(mask.float(), size=target_cam.shape[-2:], mode="area")

        outside = (1.0 - mask.float()).reshape(b, -1)

        p = F.softmax(target_cam.reshape(b, -1) / self.temperature, dim=1)

        containment = (p * outside).sum(dim=1)

        loss = self.outside_weight * containment

        # for those which have mask we have to calculate CAMLoss
        loss = loss * has_mask

        # return mean
        return loss.sum() / (has_mask.sum() + self.eps)


CAM_LOSS_REGISTRY = {
    "containment": CAMContainmentLoss,
    "squared_containment": CAMSquaredContainmentLoss,
    "allclass_squared_containment": CAMAllClassSquaredContainmentLoss,
    "containment_pointing": CAMContainmentPointingLoss,
    "softmax_containment": CAMSoftmaxContainmentLoss,
}


def build_cam_loss(config: dict) -> nn.Module:
    loss_type = str(
        config.get("cam_loss_type", "containment")
    ).lower()

    if loss_type not in CAM_LOSS_REGISTRY:
        raise ValueError(
            f"Unknown cam_loss_type='{loss_type}'. "
            f"Available options: {sorted(CAM_LOSS_REGISTRY)}"
        )

    outside_weight = float(config.get("outside_weight", 1.0))
    inside_weight = float(config.get("inside_weight", 1.0))

    if loss_type == "containment":
        return CAMContainmentLoss(
            outside_weight=outside_weight,
            inside_weight=inside_weight,
        )

    if loss_type == "squared_containment":
        return CAMSquaredContainmentLoss(
            outside_weight=outside_weight,
            inside_weight=inside_weight,
        )
        
    if loss_type == "allclass_squared_containment":
        return CAMAllClassSquaredContainmentLoss(
            outside_weight = outside_weight,
        )
    

    if loss_type == "containment_pointing":
        return CAMContainmentPointingLoss(
            outside_weight=outside_weight,
            inside_weight=inside_weight,
            margin=float(config.get("cam_pointing_margin", 0.0)),
        )

    if loss_type == "softmax_containment":
        return CAMSoftmaxContainmentLoss(
            temperature=float(
                config.get("cam_softmax_temperature", 1.0)
            ),
            outside_weight=outside_weight,
            inside_weight=inside_weight,
        )

    raise RuntimeError(f"Unhandled CAM loss type: {loss_type}")