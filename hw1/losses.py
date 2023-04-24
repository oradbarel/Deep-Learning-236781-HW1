import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """
        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1
        # SVM loss calculation based on the hinge-loss formula:
        M = torch.maximum((x_scores - x_scores.gather(1,y.unsqueeze(1)) + self.delta), torch.tensor([0]))
        loss = torch.mean(torch.sum(M, dim=1) - self.delta)
        # Save needed info for gradient calculation in self.grad_ctx:
        self.grad_ctx = [M,y,x]
        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        M, y, x = self.grad_ctx
        G = (M > 0).float()
        G[range(y.shape[0]), y] = - torch.sum(G, 1) + G[range(y.shape[0]), y]
        grad = torch.mm(torch.transpose(x, 0, 1), G) / y.shape[0]
        return grad
