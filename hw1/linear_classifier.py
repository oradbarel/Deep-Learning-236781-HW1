import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader
from hw1.losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights = torch.normal(0, weight_std, (n_features, n_classes))

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """
        class_scores = torch.mm(x, self.weights)
        y_pred = torch.argmax(class_scores, axis=1)
        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """
        acc = torch.sum(y==y_pred)/y.shape[0]
        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            epoch_train_accuracy = 0
            epoch_train_loss = 0
            
            for x_train, y_train in dl_train:
                y_pred, class_scores = self.predict(x_train)
                # calculating batch accuracy:
                current_batch_accuracy = self.evaluate_accuracy(y_train, y_pred)
                # calculate batch loss:
                current_bacth_loss = loss_fn(x_train, y_train, class_scores, y_pred) + (weight_decay / 2) * torch.norm(self.weights)
                # gradient descent step:
                self.weights -= learn_rate * loss_fn.grad()
                # update epoch train results:
                epoch_train_accuracy += current_batch_accuracy
                epoch_train_loss += current_bacth_loss

            average_epoch_train_accuracy = epoch_train_accuracy / len(dl_train)
            train_res.accuracy.append(average_epoch_train_accuracy)

            average_epoch_train_loss = epoch_train_loss / len(dl_train) 
            train_res.loss.append(average_epoch_train_loss)

            epoch_validation_accuracy = 0
            epoch_validation_loss = 0

            for x_valid, y_valid in dl_valid:
                y_pred, class_scores = self.predict(x_valid)
                # calculating batch accuracy:
                current_batch_accuracy = self.evaluate_accuracy(y_valid, y_pred)
                # calculating batch loss:
                current_bacth_loss = loss_fn(x_valid, y_valid, class_scores, y_pred) + (weight_decay / 2) * torch.norm(self.weights)
                # update epoch validation results:
                epoch_validation_accuracy += current_batch_accuracy
                epoch_validation_loss += current_bacth_loss

            average_epoch_valid_accuracy = epoch_validation_accuracy / len(dl_valid)
            valid_res.accuracy.append(average_epoch_valid_accuracy)
            average_epoch_validation_loss = epoch_validation_loss / len(dl_valid) 
            valid_res.loss.append(average_epoch_validation_loss)

            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """
        weights_for_image = self.weights[1:, :].transpose(0, 1) if has_bias else self.weights
        w_images = torch.reshape(weights_for_image, (self.n_classes,) + img_shape)
        return w_images

def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)
    hp['weight_std'] = 0.01
    hp['learn_rate'] = 0.01
    hp["weight_decay"] = 0.1
    return hp
