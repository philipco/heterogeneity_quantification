"""From
- https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_tcga_brca/loss.py#L7
- https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_ixi/loss.py"""

import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn



class CoxLoss(nn.Module):
    """Compute Cox loss given model output and ground truth (E, T)
    Parameters
    ----------
    scores: torch.Tensor, float tensor of dimension (n_samples, 1), typically
        the model output.
    truth: torch.Tensor, float tensor of dimension (n_samples, 2) containing
        ground truth event occurrences 'E' and times 'T'.
    Returns
    -------
    torch.Tensor of dimension (1, ) giving mean of Cox loss.
    """

    def __init__(self, reduction=True):
        super(CoxLoss, self).__init__()
        self.reduction  = reduction

    def forward(self, scores, truth):
        # The Cox loss calc expects events to be reverse sorted in time
        a = torch.stack((torch.squeeze(scores, dim=1), truth[:, 0], truth[:, 1]), dim=1)
        a = torch.stack(sorted(a, key=lambda a: -a[2]))
        scores = a[:, 0]
        events = a[:, 1]
        loss = torch.zeros(scores.size(0)).to(device=scores.device, dtype=scores.dtype)
        for i in range(1, scores.size(0)):
            aux = scores[: i + 1] - scores[i]
            m = aux.max()
            aux_ = aux - m
            aux_.exp_()
            loss[i] = m + torch.log(aux_.sum(0))
        loss *= events
        if self.reduction == 'none':
            return loss
        return loss.mean()


if __name__ == "__main__":

    mydataset = FedTcgaBrca(train=True, pooled=True)

    model = Baseline()

    X = torch.stack((mydataset[0][0], mydataset[1][0], mydataset[2][0]), dim=0)
    truth = torch.stack((mydataset[0][1], mydataset[1][1], mydataset[2][1]), dim=0)

    scores = model(X)

    print(X)
    print(scores)
    print(truth)

    loss = BaselineLoss()
    print(loss(scores, truth))

class DiceLoss(_Loss):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        """Get dice loss to evaluate the semantic segmentation model.
        Its value lies between 0 and 1. The more the loss is close to 0,
        the more the performance is good.

        Parameters
        ----------
        output : torch.Tensor
            Predicted values

        target : torch.Tensor
            Ground truth.

        Returns
        -------
        torch.Tensor
            A torch tensor containing the respective dice losses.
        """
        return torch.mean(1 - get_dice_score(output, target))


def get_dice_score(output, target, epsilon=1e-9):
    """Get dice score to evaluate the semantic segmentation model.
    Its value lies between 0 and 1. The more the score is close to 1,
    the more the performance is good.

    Parameters
    ----------
    output : torch.Tensor
        Predicted values

    target : torch.Tensor
        Ground truth.

    epsilon : float
        Small value to avoid zero division error.

    Returns
    -------
    torch.Tensor
        A torch tensor containing the respective dice scores.
    """
    SPATIAL_DIMENSIONS = 2, 3, 4
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return dice_score


if __name__ == "__main__":
    a = BaselineLoss()
    print(
        a(
            torch.ones((10, 1, 10, 10, 10)),
            (torch.rand((10, 1, 10, 10, 10)) > 0.5).long(),
        )
    )

