import torch.nn as nn
import torch

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channel_options,kernel_size, stride=1, padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros'):
        super(Conv2d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert isinstance(kernel_size, tuple)  and len(kernel_size) == 2
        # self.in_channels = in_channels
        # self.kernel_size = kernel_size
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.bias = nn.Parameter(torch.zeros(out_channel_options,requires_grad=True))
        self.weight = nn.Parameter(torch.zeros(out_channel_options,in_channels,*kernel_size,requires_grad=True))
        
        self.input=torch.zeros(1,in_channels,*kernel_size)
    def forward(self, x,out_channels):
        #print("out_channels",out_channels)
        bias=self.bias[out_channels]
        weight=self.weight[out_channels]
        return nn.functional.conv2d(x, weight, bias, stride=self.stride,padding= self.padding,dilation= self.dilation,groups= self.groups)

class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channel_options,kernel_size, stride=1, padding=0,output_padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros'):
        super(ConvTranspose2d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert isinstance(kernel_size, tuple)  and len(kernel_size) == 2
        
        # self.in_channels = in_channels
        # self.kernel_size = kernel_size
        
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.bias = nn.Parameter(torch.zeros(out_channel_options))
        self.weight = nn.Parameter(torch.zeros(in_channels,out_channel_options,*kernel_size))
        
        self.input=torch.zeros(1,in_channels,*kernel_size)
    def forward(self, x,out_channels):
        #print(out_channels,self.bias.shape)
        bias=self.bias[out_channels]
        weight=self.weight[:,out_channels]
        return nn.functional.conv_transpose2d(x, weight, bias,stride= self.stride,padding= self.padding,output_padding=self.output_padding,dilation= self.dilation,groups= self.groups)
# conv=Conv2d(3, 64,7)
# convtrans=ConvTranspose2d(4, 64,7)
# dummy_input=torch.zeros(1,3,32,32)
# print(convtrans(conv(dummy_input,torch.tensor((0,1,3,4))),torch.tensor((0,1,3))).shape)
