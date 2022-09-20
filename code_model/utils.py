import pickle
import numpy as np
import torch
import random
from sklearn import metrics


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# def mae_np(output, target):
#     return np.mean(np.absolute(output - target))
#
#
# def mse_np(output, target):
#     return ((output - target) ** 2).mean()


# def rmse_np(output, target):
#     return np.sqrt(((output - target) ** 2).mean())


# def mape_np(output, target):
#     output = output.flatten()
#     target = target.flatten()
#     b = np.stack((output, target), axis=0)
#     mask = (b[1] == 0)
#     b = b[:, ~mask]
#     output_mask = b[0]
#     target_mask = b[1]
#     return np.mean(np.abs((output_mask - target_mask) / target_mask)) * 100

def r2(output, target):
    return metrics.r2_score(target, output)


def mre(output, target):
    res_mre = np.average(np.abs((output - target) / target))
    if type(res_mre) is np.ndarray:
        res_mre = res_mre[0]
    return res_mre


def smape(output, target):
    res_smape = 2.0 * np.mean(np.abs(output - target) / (np.abs(output) + np.abs(target)))
    if type(res_smape) is np.ndarray:
        res_smape = res_smape[0]
    return res_smape


def mape(output, target):
    return metrics.mean_absolute_percentage_error(target, output)


def mae(output, target):
    return metrics.mean_absolute_error(target, output)


def mse(output, target):
    return metrics.mean_squared_error(target, output)


def rmse(output, target):
    rmse = np.sqrt(mse(target, output))
    if type(rmse) is np.ndarray:
        rmse = rmse[0]
    return rmse
# if __name__ == '__main__':
#     c = [1, 2, 3, 4, 5, 6, 7]
#     filename = save_variable(c, '/home/zh/temp_v')
#     d = load_variavle(filename)
#     print(d == c)
