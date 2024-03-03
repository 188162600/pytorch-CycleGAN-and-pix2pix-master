from meta import conv2d

from torch import  nn
from meta.conv2d import Conv2d,ConvTranspose2d
import  functools
from meta.section import Section
def build_conv2d(section:Section,num_options,num_pick,input_channels,output_channels,kernel_size, stride=1, padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros'):
    channel_group=output_channels//num_pick
    group_options=num_options
    print("conv","num_options",num_options,"num_pick",num_pick,"channel_group",channel_group,"group_options",group_options,"input_channels",input_channels,"output_channels",output_channels, "channel_group",channel_group!=0)
    if channel_group%num_pick!=0 or channel_group==0:
        layer=nn.Conv2d(input_channels,output_channels,kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation,groups=groups,bias=bias,padding_mode=padding_mode)
        print("layer",layer)
        section.append_layer(layer)
    else:
        layer=Conv2d(input_channels,channel_group,group_options,kernel_size, stride, padding,dilation,groups,bias,padding_mode)
        print("layer",layer)
        section.append_channelled_layer(layer)
def build_conv_transpose2d(section:Section,num_options,num_pick,input_channels,output_channels,kernel_size, stride=1, padding=0,output_padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros'):
    channel_group=output_channels//num_pick
    group_options=num_options
    print("conv_trans","num_options",num_options,"num_pick",num_pick,"channel_group",channel_group,"group_options",group_options,"input_channels",input_channels,"output_channels",output_channels, "channel_group",channel_group!=0)
    if channel_group%num_pick!=0 or channel_group==0:
        layer=nn.ConvTranspose2d(input_channels,output_channels,kernel_size=kernel_size, stride=stride, padding=padding,output_padding=output_padding,dilation=dilation,groups=groups,bias=bias,padding_mode=padding_mode)
        section.append_layer(layer)
        print("layer",layer)
    else:
        
        
        layer=ConvTranspose2d(input_channels,channel_group,group_options,kernel_size, stride, padding,output_padding,dilation,groups,bias,padding_mode)
        section.append_channelled_layer(layer)
        print("layer",layer)
    
def build_resnet_block( section:Section,num_options,num_pick,dim, padding_type, norm_layer, use_dropout, use_bias):
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
        section.append_layer(nn.ReflectionPad2d(1))
        
    elif padding_type == 'replicate':
        section.append_layer(nn.ReplicationPad2d(1))
    elif padding_type == 'zero':
        p = 1
    else:
        raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    build_conv2d(section,num_options,num_pick,dim,dim,kernel_size=3, padding=p, bias=use_bias)
    section.append_layer(norm_layer(dim))
    section.append_layer(nn.ReLU(True))
    
    if use_dropout:
        section.append_layer(nn.Dropout(0.5))
        
    p = 0
    if padding_type == 'reflect':
        section.append_layer(nn.ReflectionPad2d(1))
        #conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
        section.append_layer(nn.ReplicationPad2d(1))
        #conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
        p = 1
    else:
        raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    build_conv2d(section,num_options,num_pick,dim,dim,kernel_size=3, padding=p, bias=use_bias)
    section.append_layer(norm_layer(dim))
    #       conv_block = []
    #     p = 0
    #     if padding_type == 'reflect':
    #         conv_block += [nn.ReflectionPad2d(1)]
    #     elif padding_type == 'replicate':
    #         conv_block += [nn.ReplicationPad2d(1)]
    #     elif padding_type == 'zero':
    #         p = 1
    #     else:
    #         raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    #     conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(False)]
    #     if use_dropout:
    #         conv_block += [nn.Dropout(0.5)]

    #     p = 0
    #     if padding_type == 'reflect':
    #         conv_block += [nn.ReflectionPad2d(1)]
    #     elif padding_type == 'replicate':
    #         conv_block += [nn.ReplicationPad2d(1)]
    #     elif padding_type == 'zero':
    #         p = 1
    #     else:
    #         raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    #     conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

    #     return nn.Sequential(*conv_block)

def build_resnet_generator_section(num_options,num_pick,input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                             padding_type='reflect', *, downsamples:Section, blocks:Section, upsamples:Section):
   
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d
    downsamples.append_layer(nn.ReflectionPad2d(3))
    build_conv2d(downsamples,num_options,num_pick,input_nc,ngf,kernel_size=7, padding=0, bias=use_bias)
    downsamples.append_layer(norm_layer(ngf))
    downsamples.append_layer(nn.ReLU(True))
    
    n_downsampling = 2
    for i in range(n_downsampling):  # add downsampling layers
        mult = 2 ** i
        build_conv2d(downsamples,num_options,num_pick,ngf*mult,ngf*mult*2,kernel_size=3, stride=2, padding=1, bias=use_bias)
        downsamples.append_layer(norm_layer(ngf * mult * 2))
        downsamples.append_layer(nn.ReLU(True))
    
    
   

    mult = 2 ** n_downsampling
    for i in range(n_blocks):       # add ResNet blocks
        print("blocks",i)
        build_resnet_block(blocks,num_options,num_pick,ngf * mult, padding_type, norm_layer, use_dropout, use_bias)
    
    for i in range(n_downsampling):  # add upsampling layers
        mult = 2 ** (n_downsampling - i)
        build_conv_transpose2d(upsamples,num_options,num_pick,ngf*mult,ngf*mult//2,kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
        upsamples.append_layer(norm_layer(ngf * mult // 2))
        upsamples.append_layer(nn.ReLU(True))
    upsamples.append_layer(nn.ReflectionPad2d(3))
    build_conv2d(upsamples,num_options,num_pick,ngf, output_nc,kernel_size=7, padding=0, bias=use_bias)
    #upsamples.append_layer(nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0))
    upsamples.append_layer(nn.Tanh())
        
      


#   super(ResnetGenerator, self).__init__()
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d

#         model = [nn.ReflectionPad2d(3),
#                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
#                  norm_layer(ngf),
#                  nn.ReLU(False)]

#         n_downsampling = 2
#         for i in range(n_downsampling):  # add downsampling layers
#             mult = 2 ** i
#             model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                       norm_layer(ngf * mult * 2),
#                       nn.ReLU(False)]

#         mult = 2 ** n_downsampling
#         for i in range(n_blocks):       # add ResNet blocks

#             model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

#         for i in range(n_downsampling):  # add upsampling layers
#             mult = 2 ** (n_downsampling - i)
#             model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                                          kernel_size=3, stride=2,
#                                          padding=1, output_padding=1,
#                                          bias=use_bias),
#                       norm_layer(int(ngf * mult / 2)),
#                       nn.ReLU(False)]
#         model += [nn.ReflectionPad2d(3)]
#         model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
#         model += [nn.Tanh()]

#         self.model = nn.Sequential(*model)