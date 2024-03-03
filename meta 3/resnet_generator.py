from meta import conv2d

from torch import  nn
from meta.conv2d import Conv2d,ConvTranspose2d
import  functools
from meta.section import Section

def build_resnet_block( section,num_options,dim, padding_type, norm_layer, use_dropout, use_bias):
    """Construct a convolutional block.

    Parameters:
        dim (int)           -- the number of channels in the conv layer.
        padding_type (str)  -- the name of padding layer: reflect | replicate | zero
        norm_layer          -- normalization layer
        use_dropout (bool)  -- if use dropout layers.
        use_bias (bool)     -- if the conv layer uses bias or not

    Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
    """
   
    p = 0
    if padding_type == 'reflect':
        section.append_layer(nn.ReflectionPad2d(1),None)
        
    elif padding_type == 'replicate':
        section.append_layer(nn.ReplicationPad2d(1),None)
    elif padding_type == 'zero':
        p = 1
    else:
        raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    section.append_layer(Conv2d(dim, dim*num_options, kernel_size=3, padding=p, bias=use_bias),num_options_each_layer=dim*num_options,output_channel=dim)
    section.append_layer(norm_layer(dim),None)
    section.append_layer(nn.ReLU(False),None)
    if use_dropout:
        section.append_layer(nn.Dropout(0.5),None)
    # conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(False)]
    # if use_dropout:
    #     conv_block += [nn.Dropout(0.5)]

    p = 0
    if padding_type == 'reflect':
        section.append_layer(nn.ReflectionPad2d(1),None)
        #conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
        section.append_layer(nn.ReplicationPad2d(1),None)
        #conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
        p = 1
    else:
        raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    section.append_layer(Conv2d(dim, dim*num_options, kernel_size=3, padding=p, bias=use_bias),num_options_each_layer=dim*num_options,output_channel=dim)
    #conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

    #return nn.Sequential(*conv_block)

# def define_resnet_generator2(input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect',*,downsamples,blocks,upsamples,num_shared):
#     assert len(num_shared)==3
#     if type(norm_layer) == functools.partial:
#         use_bias = norm_layer.func == nn.InstanceNorm2d
#     else:
#         use_bias = norm_layer == nn.InstanceNorm2d

#     downsamples.extend_layers( [nn.ReflectionPad2d(3),
#                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
#                 norm_layer(ngf),
#                 nn.ReLU(False)])

#     n_downsampling = 2
#     for i in range(n_downsampling):  # add downsampling layers
#         mult = 2 ** i
#         downsamples.extend_layers( [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                     norm_layer(ngf * mult * 2),
#                     nn.ReLU(False)])


#     mult = 2 ** n_downsampling
    
#     blocks.append_layer(ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias))
#     for i in range(n_blocks-1):       # add ResNet blocks
#         blocks.append_shared_layers(0,num_shared[1])
#         #blocks += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

#     for i in range(n_downsampling):  # add upsampling layers
#         mult = 2 ** (n_downsampling - i)
#         upsamples.extend_layers( [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                                         kernel_size=3, stride=2,
#                                         padding=1, output_padding=1,
#                                         bias=use_bias),
#                     norm_layer(int(ngf * mult / 2)),
#                     nn.ReLU(False)])
#     upsamples.extend_layers( [nn.ReflectionPad2d(3)])
#     upsamples .extend_layers( [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)])
#     upsamples .extend_layers( [nn.Tanh()])
def build_resnet_generator_section(num_options_each_layer,input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                             padding_type='reflect', *, downsamples:Section, blocks:Section, upsamples:Section):
    #assert len(num_shared) == 3
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d
    downsamples.append_layer(nn.ReflectionPad2d(3),None)
    downsamples.append_layer(Conv2d(input_nc, ngf*num_options_each_layer[0], kernel_size=7, padding=0, bias=use_bias),num_options_each_layer=ngf*num_options_each_layer[0],output_channel=ngf)
    downsamples.append_layer(norm_layer(ngf*num_options_each_layer[0]),None)
    downsamples.append_layer(nn.ReLU(False),None)

    n_downsampling = 2
    for i in range(n_downsampling):  # add downsampling layers
        mult = 2 ** i
        downsamples.append_layer(Conv2d(ngf * mult, ngf * mult * 2*num_options_each_layer[1], kernel_size=3, stride=2, padding=1, bias=use_bias),num_options_each_layer=ngf * mult * 2*num_options_each_layer[0],output_channel=ngf * mult*2)
        downsamples.append_layer(norm_layer(ngf * mult * 2),None)
        downsamples.append_layer(nn.ReLU(False),None)
      
    mult = 2 ** n_downsampling
    build_resnet_block(blocks,num_options_each_layer[1],ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
    
    for i in range(n_blocks - 1):  # add ResNet blocks
        #blocks.append_shared_layers(0, num_shared[1])
        # blocks += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        build_resnet_block(blocks,num_options_each_layer[1],ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)


    for i in range(n_downsampling):  # add upsampling layers
        mult = 2 ** (n_downsampling - i)
        upsamples.append_layer(ConvTranspose2d(ngf * mult, int(ngf * mult / 2)*num_options_each_layer[2], kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),num_options_each_layer=int(ngf * mult / 2)*num_options_each_layer[2],output_channel=int(ngf * mult / 2))
        upsamples.append_layer(norm_layer(int(ngf * mult / 2)),None)
        upsamples.append_layer(nn.ReLU(False),None)
    upsamples.append_layer(nn.ReflectionPad2d(3),None)
    upsamples.append_layer(Conv2d(ngf, output_nc*num_options_each_layer[2], kernel_size=7, padding=0),num_options_each_layer=output_nc*num_options_each_layer[2],output_channel=output_nc)
    upsamples.append_layer(nn.Tanh(),None)


  