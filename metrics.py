import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

__all__ = ["accuracy", "fscore"]


def accuracy(pred: Variable, true: Variable) -> float:
    return float(accuracy_score(true, pred))


def fscore(pred: Variable, true: Variable) -> float:
    return float(f1_score(true, pred, average="weighted"))
