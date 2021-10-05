"""Contains simple ResNet-50 model for project's task."""
import timm
import torch


class PlanetClassifier(torch.nn.Module):
    """Classification model."""

    def __init__(
        self,
        name: str,
        n_classes: int,
        pretrained: bool
    ) -> None:
        """
        Parameters:
            name: Name of the architecture;
            n_classes: The number of target classes;
            pretrained: Whether to use ImageNet's weights or not.
        """
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=pretrained, num_classes=n_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map input image to raw logits.

        Parameters:
            x: Tensorized image.

        Returns:
            Raw logits.
        """
        return self.backbone(x)
