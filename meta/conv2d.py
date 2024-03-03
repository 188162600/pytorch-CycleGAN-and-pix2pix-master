import torch.nn as nn
import torch

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channel_group,out_group_options,kernel_size, stride=1, padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros'):
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
        self.bias = nn.Parameter(torch.zeros(out_group_options,out_channel_group,requires_grad=True))
        self.weight = nn.Parameter(torch.zeros(out_group_options,out_channel_group,in_channels,*kernel_size,requires_grad=True))
        self.in_channels=in_channels
        self.kernel_size=kernel_size
        self.out_channel_group=out_channel_group
        self.out_group_options=out_group_options
        #self.input=torch.zeros(1,in_channels,*kernel_size)
    def forward(self, x,out_channel_groups):
        #print("out_channels",out_channel_groups)
        bias= self.bias[out_channel_groups].view(-1)
        weight=self.weight[out_channel_groups].view(-1,self.in_channels,*self.kernel_size)
        return nn.functional.conv2d(x, weight, bias, stride=self.stride,padding= self.padding,dilation= self.dilation,groups= self.groups)
    def __repr__(self):
        return super().__repr__() +f"in_channels={self.in_channels},out_channels_groups={self.out_channel_group},out_group_options={self.out_group_options},bias={self.bias.shape},weight={self.weight.shape},kernel_size={self.kernel_size},stride={self.stride},padding={self.padding},dilation={self.dilation},groups={self.groups},padding_mode={self.padding_mode}"
class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channel_group,out_group_options,kernel_size, stride=1, padding=0,output_padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros'):
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
        self.in_channels=in_channels
        self.kernel_size=kernel_size
        self.bias = nn.Parameter(torch.zeros(out_group_options,out_channel_group))
        self.weight = nn.Parameter(torch.zeros(in_channels,out_group_options,out_channel_group,*kernel_size))
        
        self.out_channel_group=out_channel_group
        self.out_group_options=out_group_options
    def __repr__(self):
        return super().__repr__() +f"in_channels={self.in_channels},out_channels_groups={self.out_channel_group},out_group_options={self.out_group_options},bias={self.bias.shape},weight={self.weight.shape},kernel_size={self.kernel_size},stride={self.stride},padding={self.padding},dilation={self.dilation},groups={self.groups},padding_mode={self.padding_mode}"
    def forward(self, x,out_channels):
        #print(out_channels)
        bias=self.bias[out_channels].view(-1)
        weight=self.weight[:,out_channels].view(self.in_channels,-1,*self.kernel_size)
        return nn.functional.conv_transpose2d(x, weight, bias,stride= self.stride,padding= self.padding,output_padding=self.output_padding,dilation= self.dilation,groups= self.groups)
# conv=Conv2d(3,4,4,7)
# convtrans=ConvTranspose2d(8, 4,4,7)
# dummy_input=torch.zeros(1,3,32,32)
# print(convtrans(conv(dummy_input,torch.tensor((0,1))),torch.tensor((0,1))).shape)
