"""This module contains evaluation procedures."""
import torch


def f2_score(
    prob: torch.Tensor,
    label: torch.Tensor,
    threshold: float = 0.5,
    beta: float = 2
) -> float:
    """
    Compute F2-score. They use F2-score as competition metric that's why i'll
    use it too.

    Parameters:
        prob: Sigmoided tensor with predictions;
        label: GT-labels;
        threshold: Cut-off for probas;
        beta: Tradeoff parameter for F-score.

    Returns:
        Metric value.
    """
    eps = 1e-12
    prob = prob > threshold
    label = label > threshold

    tp = torch.logical_and(prob, label).float().sum(1)
    fp = torch.logical_and(
        prob, torch.logical_not(label)
    ).float().sum(1)
    fn = torch.logical_and(
        torch.logical_not(prob), label
    ).float().sum(1)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    nominator = (1 + beta**2) * precision * recall
    denominator = (beta**2 * precision + recall + eps)
    return (nominator / denominator).mean(0)
