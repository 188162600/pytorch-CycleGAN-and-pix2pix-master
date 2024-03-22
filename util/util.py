"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
def linear_interp(x:torch.Tensor, xp:torch.Tensor, fp:torch.Tensor):
    """
    Linear interpolation on PyTorch tensors.
    :param x: The x-coordinates at which to evaluate the interpolated values.
    :param xp: The x-coordinates of the data points, must be increasing.
    :param fp: The y-coordinates of the data points, same length as xp.
    :return: Interpolated values same shape as x.
    """
    # Find indices of the nearest x points, such that xp[ind-1] < x <= xp[ind]
    ind = torch.searchsorted(xp, x, right=True)
    ind = ind.clamp(min=1, max=len(xp)-1)
    xp_left = xp[ind - 1]
    xp_right = xp[ind]
    fp_left = fp[ind - 1]
    fp_right = fp[ind]

    # Linear interpolation formula
    interp_val = fp_left + (x - xp_left) * (fp_right - fp_left) / (xp_right - xp_left)
    return interp_val
def cosine_similarity_2d(x:torch.Tensor, y:torch.Tensor):
    """
    Compute the cosine similarity between two 2D tensors.
    :param a: A tensor of shape (N, M).
    :param b: A tensor of shape (N, M).
    :return: A tensor of shape (N,) containing the cosine similarity between each row of a and b.
    """

    x_norm = x / x.norm(dim=1, keepdim=True)
    y_norm = y / y.norm(dim=1, keepdim=True)

    # Compute the cosine similarity
    # Resulting shape is [16, 2]
    return  torch.mm(x_norm, y_norm.transpose(0, 1))
def confidence_loss(probability,loss):
    return torch.nn.functional.sigmoid(probability*2)*loss*1.4