from unittest import TestCase

import torch

from prediction.preprocessing.utils import haversine_tensor


class TestHaversine(TestCase):
    def setUp(self):
        red_opt = ["sum", "mean", "none"]

        self.reduction_options = [(traj_red, tensor_red) for traj_red in red_opt for tensor_red in red_opt]

    def _get_expected_shape(self, batch_size, seq_len, traj_red, tensor_red):
        if traj_red == "none":
            if tensor_red == "none":
                return (batch_size, seq_len)
            else:
                return (seq_len,)
        else:  # traj_red is 'sum' or 'mean'
            if tensor_red == "none":
                return (batch_size,)
            else:  # tensor_red is 'sum' or 'mean'
                return ()

    def test_single_trajectory_single_point(self):
        # create trajectory of shape (1, 1, 2)
        traj1 = torch.tensor([[[0, 0]]], dtype=torch.float32)
        traj2 = torch.tensor([[[0, 0]]], dtype=torch.float32)

        for traj_red, tensor_red in self.reduction_options:
            dist = haversine_tensor(traj1, traj2, traj_reduction=traj_red, tensor_reduction=tensor_red)
            self.assertEqual(dist.shape, self._get_expected_shape(1, 1, traj_red, tensor_red))
            self.assertEqual(dist.item(), 0)

    def test_single_trajectory_multiple_points(self):
        # create trajectory of shape (1, 3, 2)
        traj1 = torch.tensor([[[0, 0], [0, 1], [1, 1]]], dtype=torch.float32)
        traj2 = torch.tensor([[[0, 0], [0, 1], [1, 1]]], dtype=torch.float32)

        for traj_red, tensor_red in self.reduction_options:
            expected_shape = self._get_expected_shape(1, 3, traj_red, tensor_red)
            expected_dist = torch.zeros(expected_shape)

            dist = haversine_tensor(traj1, traj2, traj_reduction=traj_red, tensor_reduction=tensor_red)
            self.assertTrue(torch.allclose(dist, expected_dist))

    def test_multiple_trajectories_multiple_points(self):
        # create trajectories of shape (2, 3, 2)
        traj1 = torch.tensor([[[0, 0], [0, 1], [1, 1]], [[2, 2], [3, 3], [4, 4]]], dtype=torch.float32)
        traj2 = torch.tensor([[[0, 0], [0, 1], [1, 1]], [[2, 2], [3, 3], [4, 4]]], dtype=torch.float32)

        for traj_red, tensor_red in self.reduction_options:
            expected_shape = self._get_expected_shape(2, 3, traj_red, tensor_red)
            expected_dist = torch.zeros(expected_shape)

            dist = haversine_tensor(traj1, traj2, traj_reduction=traj_red, tensor_reduction=tensor_red)
            self.assertTrue(torch.allclose(dist, expected_dist))
