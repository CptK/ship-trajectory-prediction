import torch
import torch.nn as nn
from shapely.geometry import LineString
from shapely import frechet_distance

class BaselineModel(nn.Module):
    def __init__(self, pred_len: int) -> None:
        super(BaselineModel, self).__init__()
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is assumed to be a Tensor of shape (batch_size, seq_len, 2)
        last_points = x[:, -1, :]  # (batch_size, 2)
        second_last_points = x[:, -2, :]  # (batch_size, 2)

        direction = last_points - second_last_points  # (batch_size, 2)
        direction = direction.unsqueeze(1)  # (batch_size, 1, 2)

        steps = torch.arange(
            1, self.pred_len + 1, dtype=x.dtype, device=x.device
        ).view(1, -1, 1).expand(1, self.pred_len, 2)  # (1, pred_len, 2)

        last_points = last_points.unsqueeze(1) # (batch_size, 1, 2)
        y = last_points + steps * direction # (batch_size, pred_len, 2)
        return y


if __name__ == "__main__":
    x = [[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]] #, [[6, 6], [7, 7], [8, 8], [9, 9], [10, 10]]
    y = [[[6, 6], [7, 7], [8, 8], [9, 9], [10, 11]]]
    x = torch.Tensor(x)

    model = BaselineModel(5)
    y_pred = model.forward(x)

    y_string = LineString(y[0])
    y_pred_string = LineString(y_pred[0].numpy())
    print(y_string)
    print(y_pred_string)
    print(frechet_distance(y_string, y_pred_string))
    

    print(y)
