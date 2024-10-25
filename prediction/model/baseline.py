import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(self, pred_len: int) -> None:
        super().__init__()
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is assumed to be a Tensor of shape (batch_size, seq_len, 2)
        last_points = x[:, -1, :]  # (batch_size, 2)
        second_last_points = x[:, -2, :]  # (batch_size, 2)

        direction = last_points - second_last_points  # (batch_size, 2)
        direction = direction.unsqueeze(1)  # (batch_size, 1, 2)

        steps = (
            torch.arange(1, self.pred_len + 1, dtype=x.dtype, device=x.device)
            .view(1, -1, 1)
            .expand(1, self.pred_len, 2)
        )  # (1, pred_len, 2)

        last_points = last_points.unsqueeze(1)  # (batch_size, 1, 2)
        y = last_points + steps * direction  # (batch_size, pred_len, 2)
        return y
