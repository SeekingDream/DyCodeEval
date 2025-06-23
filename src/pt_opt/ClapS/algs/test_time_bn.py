import torch
from typing import Optional, Literal

PossibleCentralTendencyEstimators = Literal["mean", "median"]


class BatchNormCalibrate(torch.nn.Module):
    def __init__(
        self,
        momentum: float = 0.1,
        central_tendency_estmator: PossibleCentralTendencyEstimators = "mean",
        demean_only: bool = False,
    ) -> None:
        super().__init__()
        self.running_mean: Optional[torch.Tensor] = None
        self.running_variance: Optional[torch.Tensor] = None
        self.momentum: float = momentum
        self.training: bool = False
        self.central_tendency_estimator: PossibleCentralTendencyEstimators = (
            central_tendency_estmator
        )
        self.demean_only = demean_only

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def forward(self, logits: torch.Tensor, flush: bool = False) -> torch.Tensor:
        """
        Args:
            logits: torch.Tensor of shape (... x N x d) where `d` is the dimensionality of the logits
                (in this case, the number of classes), `N` is the number of samples.
            flush: whether to reset the running mean and variance.

        Returns:
            standardized_logits: when in training, the running stats are tracked but the origial
                logits are returned. torch.Tensor of shape (... x N x d)
                When in eval mode, the running stats are fixed and the logits are standardized
                along the final dimension. return Tensor of shape (... x N x d)
        """
        if (self.running_mean is None or self.running_variance is None) and (
            not self.training
        ):
            raise ValueError(
                "running_mean and/or running_variance is not defined -- cannot run eval mode now."
            )
        if self.training:
            logits_mean = (
                torch.mean(logits, dim=-2, keepdim=True)
                if self.central_tendency_estimator == "mean"
                else torch.median(logits, dim=-2, keepdim=True)
            )
            logits_variance = torch.var(logits, dim=-2, keepdim=True)
            if self.running_mean is None or self.running_variance is None or flush:
                self.running_mean, self.running_variance = logits_mean, logits_variance
            else:
                assert self.central_tendency_estimator != "median"
                # median is not a linear operation and running median cannot be estimated by simply tracking
                # the statistics.
                self.running_mean = (
                    1.0 - self.momentum
                ) * self.running_mean + logits_mean * self.momentum
                self.running_variance = (
                    1.0 - self.momentum
                ) * self.running_variance + logits_variance * self.momentum
            print(
                f"Current running_mean: {self.running_mean.numpy()}; running variance: {self.running_variance.numpy()}"
            )
            return logits
        else:
            demeaned_logits = logits - self.running_mean
            if self.demean_only:
                return demeaned_logits
            return demeaned_logits / torch.sqrt(self.running_variance)
