from __future__ import annotations

import argparse
import unittest

from methods.Ours.core import AdaptiveGuidanceController


def controller_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "beta_schedule": "alg",
        "beta_on": 2.5,
        "alg_threshold": -0.02,
        "alg_smoothing_window": 50,
        "guidance_stop_epoch": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def reference_derivatives(
    losses: list[float], window: int
) -> tuple[list[float], list[float]]:
    raw: list[float] = []
    smoothed: list[float] = []
    for index, current in enumerate(losses):
        epoch = index + 1
        if epoch == 1:
            derivative = 0.0
        elif epoch <= window:
            derivative = (current - sum(losses[:index]) / index) / epoch
        else:
            derivative = (current - losses[index - window]) / window
        raw.append(derivative)
        recent = raw[max(0, len(raw) - window) :]
        smoothed.append(sum(recent) / len(recent))
    return raw, smoothed


class AdaptiveGuidanceControllerTest(unittest.TestCase):
    def test_matches_alg_equations(self) -> None:
        losses = [10.0 - 0.1 * epoch for epoch in range(120)]
        expected_raw, expected_smoothed = reference_derivatives(losses, 50)
        controller = AdaptiveGuidanceController(
            controller_args(alg_threshold=1.0)
        )
        for epoch, loss in enumerate(losses, 1):
            self.assertEqual(controller.beta_for_epoch(epoch), 2.5)
            controller.observe(epoch, loss)

        for actual, expected in zip(
            controller.derivative_history, expected_raw, strict=True
        ):
            self.assertAlmostEqual(actual, expected, places=12)
        for actual, expected in zip(
            controller.smoothed_derivative_history,
            expected_smoothed,
            strict=True,
        ):
            self.assertIsNotNone(actual)
            self.assertAlmostEqual(float(actual), expected, places=12)

    def test_crossing_epoch_is_last_guided_epoch(self) -> None:
        controller = AdaptiveGuidanceController(
            controller_args(alg_smoothing_window=3)
        )
        losses = [4.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        for epoch, loss in enumerate(losses, 1):
            beta = controller.beta_for_epoch(epoch)
            if beta == 0.0:
                break
            controller.observe(epoch, loss)

        self.assertIsNotNone(controller.stop_epoch)
        assert controller.stop_epoch is not None
        self.assertEqual(controller.beta_history[controller.stop_epoch - 1], 2.5)
        self.assertEqual(controller.beta_for_epoch(controller.stop_epoch + 1), 0.0)

    def test_epoch_one_cannot_stop_guidance(self) -> None:
        controller = AdaptiveGuidanceController(controller_args())
        self.assertEqual(controller.beta_for_epoch(1), 2.5)
        controller.observe(1, 4.0)
        self.assertTrue(controller.active)
        self.assertIsNone(controller.stop_epoch)


if __name__ == "__main__":
    unittest.main()
