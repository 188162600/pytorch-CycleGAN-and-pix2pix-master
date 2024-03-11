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
        #print(classname,"classname",classname.find('Conv'),classname.find('Conv2d'))
        # if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            
        #     if init_type == 'normal':
        #         init.normal_(m.weight.data, 0.0, init_gain)
        #     elif init_type == 'xavier':
        #         init.xavier_normal_(m.weight.data, gain=init_gain)
        #     elif init_type == 'kaiming':
        #         init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        #     elif init_type == 'orthogonal':
        #         init.orthogonal_(m.weight.data, gain=init_gain)
        #     else:
        #         raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        #     if hasattr(m, 'bias') and m.bias is not None:
        #         init.constant_(m.bias.data, 0.0)
        # elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        #     init.normal_(m.weight.data, 1.0, init_gain)
        #     init.constant_(m.bias.data, 0.0)
        #print(classname)
        if isinstance(m, (Conv2d,ConvTranspose2d)):
            #print(m.weight.shape,m.bias.shape)
            if init_type == 'normal':
                m.init_weight(lambda x: init.normal_(x, 0.0, init_gain))
               
            elif init_type == 'xavier':
                m.init_weight(lambda x: init.xavier_normal_(x, gain=init_gain))
                
            elif init_type == 'kaiming':
                m.init_weight(lambda x: init.kaiming_normal_(x, a=0, mode='fan_in'))
            elif init_type == 'orthogonal':
                m.init_weight(lambda x: init.orthogonal_(x, gain=init_gain))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif hasattr(m, 'weight')  and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            
            if init_type == 'normal' :
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
<<<<<<< Updated upstream
    # if len(gpu_ids) > 0:
    #     print("gpu_ids",gpu_ids)
    #     assert(torch.cuda.is_available())
    #     net.to(gpu_ids[0])
    #     net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net
def define_resnet_section(num_options,num_pick,input_nc, output_nc, ngf, norm, use_dropout=False, n_blocks=6, padding_type='reflect', init_type='normal', init_gain=0.02, gpu_ids=[]):
    downsample= Section("downsample",num_options,num_pick)
    blocks= Section("resnet_blocks",num_options,num_pick)
    upsample= Section("upsample",num_options,num_pick)
    norm=get_norm_layer(norm)
    build_resnet_generator_section(num_options,num_pick,input_nc, output_nc, ngf, norm, use_dropout, n_blocks, padding_type,downsamples= downsample,blocks= blocks,upsamples=upsample)
    #for child in downsample.children():
    for layers in downsample.base_layers:
=======
    if len(gpu_ids) > 0:
        print("gpu_ids",gpu_ids)
        assert(torch.cuda.is_available())
        #net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net
def define_resnet_generator2(input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect',*,downsamples,blocks,upsamples,num_shared):
    assert len(num_shared)==3
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    downsamples.extend_layers( [nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                norm_layer(ngf),
                nn.ReLU(True)])

    n_downsampling = 2
    for i in range(n_downsampling):  # add downsampling layers
        mult = 2 ** i
        downsamples.extend_layers( [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True)])


    mult = 2 ** n_downsampling
    
    blocks.append_layer(ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias))
    for i in range(n_blocks-1):       # add ResNet blocks
        blocks.append_shared_layers(0,num_shared[1])
        #blocks += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

    for i in range(n_downsampling):  # add upsampling layers
        mult = 2 ** (n_downsampling - i)
        upsamples.extend_layers( [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)])
    upsamples.extend_layers( [nn.ReflectionPad2d(3)])
    upsamples .extend_layers( [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)])
    upsamples .extend_layers( [nn.Tanh()])
def define_resnet_sections2(num_options_each_layer,num_shared,input_nc, output_nc, ngf, norm, use_dropout=False, n_blocks=6, padding_type='reflect', init_type='normal', init_gain=0.02, gpu_ids=[]):
    assert len(num_options_each_layer)==3
    downsample_layers=[]
    resnet_blocks=[]
    upsample_layers=[]
    downsample_section=Section("downsample",downsample_layers,num_options_each_layer[0])
    resnet_section= Section("resnet",resnet_blocks,num_options_each_layer[1])
    upsample_section= Section("upsample",upsample_layers,num_options_each_layer[2])
    define_resnet_generator2(input_nc, output_nc, ngf, get_norm_layer(norm_type=norm) , use_dropout, n_blocks, padding_type,downsamples= downsample_section,blocks= resnet_section,upsamples=upsample_section,num_shared=num_shared)
    sections= [downsample_section,resnet_section,upsample_section]
    # for section in sections:
    #     for layer in section.base_layers:
    #         init_net(layer, init_type, init_gain, gpu_ids)
    return sections
def define_resnet_generator(input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect',*,downsamples,blocks,upsamples):
    
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        downsamples+=  [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            downsamples += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]


        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            blocks += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            upsamples += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        upsamples += [nn.ReflectionPad2d(3)]
        upsamples += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        upsamples += [nn.Tanh()]
def define_resnet_sections(num_options_each_layer,input_nc, output_nc, ngf, norm, use_dropout=False, n_blocks=6, padding_type='reflect', init_type='normal', init_gain=0.02, gpu_ids=[]):
    downsample_layers=[]
    resnet_blocks=[]
    upsample_layers=[]
    define_resnet_generator(input_nc, output_nc, ngf, get_norm_layer(norm_type=norm) , use_dropout, n_blocks, padding_type,downsamples= downsample_layers,blocks= resnet_blocks,upsamples=upsample_layers)
    sections= [Section("downsample",downsample_layers,num_options_each_layer),Section("resnet",resnet_blocks,num_options_each_layer),Section("upsample",upsample_layers,num_options_each_layer)]
    for section in sections:
        init_net(section, init_type, init_gain, gpu_ids)
    return sections

# def define_unet_generator(input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,*,downsamples,blocks,upsamples):
#         downsamples+= [UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)]  # add the innermost layer
>>>>>>> Stashed changes
        
        init_net(layers, init_type, init_gain, gpu_ids)
    for layers in blocks.base_layers:
        
        init_net(layers, init_type, init_gain, gpu_ids)
    for layers in upsample.base_layers:
        
        init_net(layers, init_type, init_gain, gpu_ids)   
   
    # init_net(downsample, init_type, init_gain, gpu_ids)
    # init_net(blocks, init_type, init_gain, gpu_ids)
    # init_net(upsample, init_type, init_gain, gpu_ids)
    # for section in [downsample,blocks,upsample]:
    #     print(section.name)
    #     for i,layer in enumerate( section.base_layers):
    #         if isinstance(layer,(Conv2d,ConvTranspose2d)):
    #             print(layer.weight.shape,layer.bias.shape,section.output_channels[i])


    return [downsample,blocks,upsample]

def define_resnet_encoder(name,**kwargs):
    return find_resnet_encoder(name)(**kwargs)
