from __future__ import annotations

import unittest

import torch

from methods.Ours.ours import Ours


def inputs() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    student = [
        torch.randn(1, 192, 14, 14, requires_grad=True) for _ in range(12)
    ]
    teacher = [
        torch.randn(1, 16, 32, 32),
        torch.randn(1, 32, 16, 16),
        torch.randn(1, 64, 8, 8),
    ]
    return student, teacher


class OursGridTest(unittest.TestCase):
    def test_paper_mode_uses_teacher_stage_grids(self) -> None:
        student, teacher = inputs()
        module = Ours(grid_resize_mode="teacher")
        alignment, fusion, aligned, fused, targets = module(student, teacher)
        expected = [(1, 16, 32, 32), (1, 32, 16, 16), (1, 64, 8, 8)]
        self.assertEqual([tuple(value.shape) for value in aligned], expected)
        self.assertEqual([tuple(value.shape) for value in fused], expected)
        self.assertEqual([tuple(value.shape) for value in targets], expected)
        (alignment + fusion).backward()
        self.assertIsNotNone(student[0].grad)

    def test_source_compatibility_mode_uses_larger_grid(self) -> None:
        student, teacher = inputs()
        module = Ours(grid_resize_mode="larger")
        _, _, aligned, _, targets = module(student, teacher)
        expected = [(1, 16, 32, 32), (1, 32, 16, 16), (1, 64, 14, 14)]
        self.assertEqual([tuple(value.shape) for value in aligned], expected)
        self.assertEqual([tuple(value.shape) for value in targets], expected)


if __name__ == "__main__":
    unittest.main()
