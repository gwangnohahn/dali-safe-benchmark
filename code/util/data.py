"""Dataset utilities using the paper's convention (:math:`y`=performance, :math:`x`=condition).

Tensor attributes keep historical names: ``Dataset.x`` stores performance :math:`y` values,
while ``Dataset.y`` stores condition :math:`x` values.
"""

import torch
import pandas
import numpy
import torch.utils.data
from copy import deepcopy
from torch.nn.functional import softmax

# Select compute device automatically for tensor operations.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Data:
    """Lightweight container for normalized performance/condition tensors."""
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset bundling normalized performance (:math:`y`) and condition (:math:`x`)."""
    def __init__(self, x, y, x_mean, x_std, y_mean, y_std):
        self.x = x
        self.y = y.view(x.shape[0], -1)
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.data = torch.cat([self.x, self.y], dim=1)
        self.x_cuda = deepcopy(self.x).to(device)
        self.y_cuda = deepcopy(self.y).to(device)

    def __len__(self):
        """Return the number of performance/condition pairs."""
        return self.x.shape[0]

    def __getitem__(self, idx):
        """Return the normalized performance/condition pair at ``idx``."""
        return self.x[idx], self.y[idx]

    def get_x_bounds(self):
        """Lower/upper bounds of normalized performances :math:`y`."""
        lb, _ = torch.min(self.x, dim=0)
        ub, _ = torch.max(self.x, dim=0)

        return lb.numpy(), ub.numpy()

    def get_ref_target(self, x, train):
        """Retrieve reference conditions :math:`x` via distance weighting around ``x`` (performance)."""
        if train:
            _, idx = torch.topk(-torch.cdist(x.to(device), self.x_cuda), k=5)
            ref_y = torch.mean(self.y_cuda[idx[:, 1:]], dim=1)
        else:
            # _, idx = torch.topk(-torch.cdist(x, self.x_cuda), k=3)
            # ref_y = torch.mean(self.y_cuda[idx], dim=1)
            _, idx = torch.topk(-torch.cdist(x.to(device), self.x_cuda), k=5)
            ref_y = torch.mean(self.y_cuda[idx[:, 1:]], dim=1)

        vals, idx = torch.topk(-torch.cdist(x.to(device), self.x_cuda), k=5)
        coeff = softmax(torch.exp(100 * vals[:, 1:]), dim=1)
        ref_y = torch.sum(coeff.unsqueeze(2) * self.y_cuda[idx[:, 1:]], dim=1)

        ref_min, _ = torch.min(self.y_cuda[idx[:, 1:]], dim=1)
        ref_max, _ = torch.max(self.y_cuda[idx[:, 1:]], dim=1)
        ref_y = torch.cat([ref_min, ref_max], dim=1)

        return ref_y

    def get_k_folds(self, num_folds, random_seed=None):
        """Yield ``num_folds`` train/test splits that preserve performance/condition scaling."""
        if random_seed is not None:
            numpy.random.seed(random_seed)

        num_feats = self.x.shape[1]
        idx_rand = numpy.array_split(numpy.random.permutation(len(self)), num_folds)
        sub_datasets = list()
        for i in range(0, num_folds):
            sub_datasets.append(self.data[torch.tensor(idx_rand[i], dtype=torch.long)])

        k_folds = list()
        for i in range(0, num_folds):
            data_train = torch.cat(sub_datasets[:i] + sub_datasets[i+1:], dim=0)
            data_test = sub_datasets[i]
            dataset_train = Dataset(data_train[:, :num_feats], data_train[:, num_feats:],
                                    self.x_mean, self.x_std, self.y_mean, self.y_std)
            dataset_test = Dataset(data_test[:, :num_feats], data_test[:, num_feats:],
                                   self.x_mean, self.x_std, self.y_mean, self.y_std)
            k_folds.append([dataset_train, dataset_test])

        return k_folds

    def split_by_target(self, thr):
        """Split the dataset by a scalar threshold applied to conditions :math:`x`."""
        y = self.y_std * self.y.numpy() + self.y_mean
        idx1 = list()
        idx2 = list()

        for i in range(0, len(self)):
            if y[i] < thr:
                idx1.append(i)
            else:
                idx2.append(i)
        idx1 = numpy.array(idx1, dtype=int)
        idx2 = numpy.array(idx2, dtype=int)

        dataset_train = Dataset(self.x[idx1], self.y[idx1], self.x_mean, self.x_std, self.y_mean, self.y_std)
        dataset_test = Dataset(self.x[idx2], self.y[idx2], self.x_mean, self.x_std, self.y_mean, self.y_std)

        return dataset_train, dataset_test


def load_dataset(path_dataset, idx_input_vars, idx_output_vars):
    """Load Excel data and normalize to performance/condition tensors."""
    _idx_input_vars = numpy.atleast_1d(idx_input_vars)
    _idx_output_vars = numpy.atleast_1d(idx_output_vars)
    data = numpy.array(pandas.read_excel(path_dataset))

    x_mean = numpy.mean(data[:, _idx_input_vars], axis=0).reshape(1, -1)
    x_std = numpy.std(data[:, _idx_input_vars], axis=0).reshape(1, -1)
    x_norm = (data[:, _idx_input_vars] - x_mean) / x_std

    y_mean = numpy.mean(data[:, _idx_output_vars], axis=0).reshape(1, -1)
    y_std = numpy.std(data[:, _idx_output_vars], axis=0).reshape(1, -1)
    y_norm = (data[:, _idx_output_vars] - y_mean) / y_std

    return Dataset(torch.tensor(x_norm, dtype=torch.float),
                   torch.tensor(y_norm, dtype=torch.float),
                   x_mean, x_std, y_mean, y_std)
