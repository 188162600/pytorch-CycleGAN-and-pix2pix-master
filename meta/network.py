import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from meta.task import Task
from meta.section import Section
from meta.resnet_generator import build_resnet_generator_section
from meta.resnet_encoder import find_resnet_encoder

from meta.conv2d import Conv2d,ConvTranspose2d
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        print("gpu_ids",gpu_ids)
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net
def define_resnet_section(num_options,num_pick,input_nc, output_nc, ngf, norm, use_dropout=False, n_blocks=6, padding_type='reflect', init_type='normal', init_gain=0.02, gpu_ids=[]):
    downsample= Section("downsample",num_options,num_pick)
    blocks= Section("resnet_blocks",num_options,num_pick)
    upsample= Section("upsample",num_options,num_pick)
    norm=get_norm_layer(norm)
    build_resnet_generator_section(num_options,num_pick,input_nc, output_nc, ngf, norm, use_dropout, n_blocks, padding_type,downsamples= downsample,blocks= blocks,upsamples=upsample)
    init_net(downsample, init_type, init_gain, gpu_ids)
    init_net(blocks, init_type, init_gain, gpu_ids)
    init_net(upsample, init_type, init_gain, gpu_ids)
    # for section in [downsample,blocks,upsample]:
    #     print(section.name)
    #     for i,layer in enumerate( section.base_layers):
    #         if isinstance(layer,(Conv2d,ConvTranspose2d)):
    #             print(layer.weight.shape,layer.bias.shape,section.output_channels[i])


    return [downsample,blocks,upsample]

def define_resnet_encoder(name,**kwargs):
    return find_resnet_encoder(name)(**kwargs)
