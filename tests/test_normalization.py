"""Regression checks for input/visualization/attack normalization.

Run: python -B tests/test_normalization.py
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adversarial import adversarial_accuracy, fgsm_attack, pgd_attack
from src.data import get_dataloaders
from src.losses import CAMSoftmaxContainmentLoss
from src.normalization import (
    BIT_MEAN, BIT_STD, IMAGENET_MEAN, IMAGENET_STD,
    configure_normalization, denormalize_images, loader_normalization,
    normalization_for_loader, normalize_images, resolve_normalization,
    validate_checkpoint_normalization, validate_normalization,
)
from src.reports.snapshots import CAMSnapshotter, _denormalize
from src.train import run_training


class TinyCAM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cam_conv = torch.nn.Conv2d(3, 2, 1)

    def forward(self, x):
        cams = self.cam_conv(x)
        return cams.mean((2, 3)), cams


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        (ROOT / 'tmp').mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix='normalization_test_', dir=ROOT / 'tmp')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        self.pixels[0, 0] = [0, 128, 255]
        self.mask = np.zeros((8, 8), dtype=np.uint8)
        self.mask[2:6, 1:7] = 255
        self.mask_dirs = {}
        for label in ('a', 'b'):
            directory = self.root / 'images' / label
            directory.mkdir(parents=True)
            masks = self.root / 'masks' / label
            masks.mkdir(parents=True)
            self.mask_dirs[label] = str(masks)
            for i in range(5):
                Image.fromarray(self.pixels).save(directory / f'{i}.png')
                Image.fromarray(self.mask).save(masks / f'{i}.png')

    def loaders(self, mean=BIT_MEAN, std=BIT_STD):
        with contextlib.redirect_stdout(io.StringIO()):
            return get_dataloaders(
                str(self.root / 'images'), 8, 2, 0.2, 42, 0,
                mask_dirs=self.mask_dirs, test_ratio=0.2,
                normalize_mean=mean, normalize_std=std,
            )

    def test_defaults_match_pretrained_model_metadata(self):
        import timm
        from torchvision.models import ResNet18_Weights
        bit = timm.get_pretrained_cfg('resnetv2_50x1_bit.goog_in21k')
        self.assertEqual(resolve_normalization({'model_type': 'adapter_bit'}), (bit.mean, bit.std))
        resnet = ResNet18_Weights.DEFAULT.transforms()
        self.assertEqual(resolve_normalization({'model_type': 'adapter'}), (tuple(resnet.mean), tuple(resnet.std)))
        self.assertEqual(resolve_normalization({'model_type': 'adapter_clip'}), (BIT_MEAN, BIT_STD))

    def test_explicit_legacy_stats_and_resolved_config(self):
        config = {'model_type': 'adapter_bit', 'normalize_mean': IMAGENET_MEAN, 'normalize_std': IMAGENET_STD}
        self.assertEqual(configure_normalization(config), (IMAGENET_MEAN, IMAGENET_STD))
        self.assertEqual(config['normalize_mean'], list(IMAGENET_MEAN))
        self.assertEqual(configure_normalization({'model_type': 'adapter_bit'}), (BIT_MEAN, BIT_STD))

    def test_invalid_and_conflicting_statistics_rejected(self):
        for mean, std in [(None, BIT_STD), (BIT_MEAN, None), ([0, 0], BIT_STD),
                          (BIT_MEAN, [0, 1, 1]), (BIT_MEAN, [-1, 1, 1]),
                          ([float('nan'), 0, 0], BIT_STD), (BIT_MEAN, [float('inf'), 1, 1])]:
            with self.subTest(mean=mean, std=std), self.assertRaises(ValueError):
                validate_normalization(mean, std)
        with self.assertRaises(ValueError):
            resolve_normalization({'normalize_mean': BIT_MEAN})
        with self.assertRaises(ValueError):
            self.loaders(BIT_MEAN, None)
        with self.assertRaisesRegex(ValueError, 'attack_normalize'):
            resolve_normalization({'model_type': 'adapter_bit', 'attack_normalize_mean': IMAGENET_MEAN,
                                   'attack_normalize_std': IMAGENET_STD})

    def test_all_splits_use_selected_stats_and_leave_masks_binary(self):
        for mean, std in [(BIT_MEAN, BIT_STD), (IMAGENET_MEAN, IMAGENET_STD)]:
            loaders = self.loaders(mean, std)[:3]
            expected = torch.from_numpy(self.pixels.copy()).permute(2, 0, 1).float() / 255
            expected = (expected - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
            for loader in loaders:
                self.assertEqual(loader_normalization(loader), (mean, std))
                batch = next(iter(loader))
                torch.testing.assert_close(batch['image'][0], expected)
                torch.testing.assert_close(batch['mask'][0, 0], torch.from_numpy(self.mask / 255).float())
                self.assertEqual(batch['has_mask'].tolist(), [1.0, 1.0])

    def test_normalization_round_trip_and_bit_endpoints(self):
        pixels = torch.tensor([0., .5, 1.]).view(3, 1, 1).expand(3, 5, 4)
        torch.testing.assert_close(normalize_images(pixels, BIT_MEAN, BIT_STD)[:, 0, 0], torch.tensor([-1., 0., 1.]))
        for stats in [(BIT_MEAN, BIT_STD), (IMAGENET_MEAN, IMAGENET_STD)]:
            for x in [pixels, pixels.unsqueeze(0)]:
                torch.testing.assert_close(denormalize_images(normalize_images(x, *stats), *stats), x, atol=1e-7, rtol=0)
        with self.assertRaises(ValueError):
            normalize_images(pixels, BIT_MEAN, None)

    def test_snapshots_infer_actual_loader_stats_and_reconstruct_rgb(self):
        loader = self.loaders()[1]
        snapshot = CAMSnapshotter(loader, 'cpu', ['a', 'b'], self.root, num_samples=1)
        sample = snapshot.samples[0]
        restored = _denormalize(sample['image'], snapshot.normalize_mean, snapshot.normalize_std)
        self.assertLessEqual(np.abs(restored.astype(int) - self.pixels.astype(int)).max(), 1)
        rendered_path = snapshot.capture(TinyCAM(), epoch=1)
        with Image.open(rendered_path) as rendered:
            input_tile = np.array(rendered)[22:30, :8]
        self.assertLessEqual(np.abs(input_tile.astype(int) - self.pixels.astype(int)).max(), 1)
        with self.assertRaisesRegex(ValueError, 'Normalization mismatch'):
            CAMSnapshotter(loader, 'cpu', ['a', 'b'], self.root,
                           normalize_mean=IMAGENET_MEAN, normalize_std=IMAGENET_STD)

    def test_attack_zero_epsilon_identity_and_pixel_budget(self):
        model = TinyCAM().eval()
        pixels = torch.rand(2, 3, 8, 8)
        pixels[0, :, 0, 0] = torch.tensor([0., .5, 1.])
        labels = torch.tensor([0, 1])
        for mean, std in [(BIT_MEAN, BIT_STD), (IMAGENET_MEAN, IMAGENET_STD)]:
            images = normalize_images(pixels, mean, std)
            for epsilon in [0., 2 / 255]:
                attacks = [fgsm_attack(model, images, labels, epsilon, mean, std),
                           pgd_attack(model, images, labels, epsilon, 1 / 255, 2, mean=mean, std=std)]
                for output in attacks:
                    recovered = denormalize_images(output, mean, std)
                    self.assertLessEqual((recovered - pixels).abs().max().item(), epsilon + 2e-7)
                    if epsilon == 0:
                        torch.testing.assert_close(output, images, atol=3e-7, rtol=0)
                        torch.testing.assert_close(model(output)[0], model(images)[0])

    def test_attack_evaluation_infers_loader_stats_and_rejects_mismatch(self):
        loader = self.loaders()[2]
        seen = []
        def capture_attack(model, images, labels, mean, std):
            seen.append((mean, std))
            return images
        adversarial_accuracy(TinyCAM(), loader, 'cpu', capture_attack)
        self.assertEqual(seen, [(BIT_MEAN, BIT_STD)])
        with self.assertRaisesRegex(ValueError, 'Normalization mismatch'):
            adversarial_accuracy(TinyCAM(), loader, 'cpu', capture_attack, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def test_checkpoint_metadata_and_legacy_warning(self):
        checkpoint = {'input_normalization': {'mean': list(BIT_MEAN), 'std': list(BIT_STD)}}
        validate_checkpoint_normalization(checkpoint, BIT_MEAN, BIT_STD)
        with self.assertRaisesRegex(ValueError, 'Checkpoint normalization'):
            validate_checkpoint_normalization(checkpoint, IMAGENET_MEAN, IMAGENET_STD)
        with self.assertWarnsRegex(UserWarning, 'older src/data.py hardcoded ImageNet'):
            validate_checkpoint_normalization({}, IMAGENET_MEAN, IMAGENET_STD)

    def test_training_saves_actual_normalization_in_checkpoint_and_config(self):
        train, val, test, classes = self.loaders()
        model = TinyCAM()
        kwargs = dict(model=model, train_loader=train, val_loader=val, test_loader=test,
                      classes=classes, ce_loss_fn=torch.nn.CrossEntropyLoss(),
                      cam_loss_fn=CAMSoftmaxContainmentLoss(), optimizer=torch.optim.SGD(model.parameters(), lr=.01),
                      scheduler=None, scaler=None, device='cpu', lambda_cam=.1, use_amp=False, epochs=1,
                      runs_dir=str(self.root / 'runs'))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_training(**kwargs, config={'model_type': 'adapter_bit', 'diagnostics': False, 'cam_debug': False})
        checkpoint = torch.load(result['run_dir'] / 'last_resnet18_cam.pth', weights_only=True)
        validate_checkpoint_normalization(checkpoint, BIT_MEAN, BIT_STD)
        self.assertEqual(checkpoint['input_normalization']['mean'], list(BIT_MEAN))
        config = yaml.safe_load((result['run_dir'] / 'config.yaml').read_text())
        self.assertEqual(config['normalize_std'], list(BIT_STD))
        info = json.loads((result['run_dir'] / 'model_info.json').read_text())
        self.assertEqual(info['input_normalization'], checkpoint['input_normalization'])
        import erf_suite
        with contextlib.redirect_stdout(io.StringIO()):
            loaded = erf_suite.load_explicit_checkpoint(TinyCAM(), result['run_dir'] / 'last_resnet18_cam.pth',
                                                        'cpu', 'model', normalization=(BIT_MEAN, BIT_STD))
        torch.testing.assert_close(loaded.cam_conv.weight, model.cam_conv.weight)
        with self.assertRaisesRegex(ValueError, 'Normalization mismatch'):
            run_training(**kwargs, config={'model_type': 'plain'})

    def test_main_and_sweep_forward_stats_before_training(self):
        import main
        import sweep
        config = {'data_dir': str(self.root / 'images'), 'model_type': 'adapter_bit', 'pretrained': False}
        path = self.root / 'config.yaml'
        path.write_text(yaml.safe_dump(config))
        loaders = self.loaders()
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(main, 'CONFIG_PATH', path), \
                mock.patch.object(main, 'get_device', return_value='cpu'), \
                mock.patch.object(main, 'get_dataloaders', return_value=loaders) as build, \
                mock.patch.object(main, 'build_model', return_value=TinyCAM()), \
                mock.patch.object(main, 'run_training') as train:
            main.main()
            self.assertEqual(build.call_args.kwargs['normalize_mean'], BIT_MEAN)
            self.assertEqual(train.call_args.kwargs['config']['normalize_std'], list(BIT_STD))
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(sweep, 'get_device', return_value='cpu'), \
                mock.patch.object(sweep, 'get_dataloaders', return_value=loaders) as build, \
                mock.patch.object(sweep, 'build_model', return_value=TinyCAM()), \
                mock.patch.object(sweep, 'run_training', return_value={'run_dir': self.root, 'best_val_acc': .5,
                                  'history': {'train_acc': [.5], 'val_acc': [.5]}}):
            sweep.run_single(config, 1, .1, 0)
            self.assertEqual(build.call_args.kwargs['normalize_std'], list(BIT_STD))

    def test_erf_resolves_nested_config_and_rejects_incompatible_model_input(self):
        import erf_suite
        self.assertEqual(erf_suite.infer_norm({'model_type': 'adapter_bit'}), (BIT_MEAN, BIT_STD))
        self.assertEqual(erf_suite.infer_norm({'data': {'normalize_mean': BIT_MEAN, 'normalize_std': BIT_STD}}),
                         (BIT_MEAN, BIT_STD))
        with self.assertRaisesRegex(ValueError, 'shared ERF loader'):
            erf_suite.build_one_model({'model_type': 'plain'}, 2, 'cpu', None, 'teacher',
                                      input_normalization=(BIT_MEAN, BIT_STD))

    def test_comparison_renderer_round_trip_and_current_cam_api(self):
        from src.compare_cams import denormalize_image, get_target_cam
        batch = next(iter(self.loaders()[1]))
        restored = denormalize_image(batch['image'][0], BIT_MEAN, BIT_STD)
        self.assertLessEqual(np.abs(restored.astype(int) - self.pixels.astype(int)).max(), 1)
        cam = get_target_cam(TinyCAM().eval(), batch['image'][:1], batch['label'][:1], 'cpu')
        self.assertEqual(cam.shape, (8, 8))
        self.assertTrue(np.isfinite(cam).all())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
