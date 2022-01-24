import math
import collections
from collections import OrderedDict
from itertools import repeat
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from math import cos, pi, sqrt

class BasicConv(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias_attr=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias_attr=bias_attr)
        self.bn = nn.BatchNorm2D(out_planes,epsilon=1e-5, momentum=0.01) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Layer):
    def forward(self, x):
        return paddle.reshape(x, [x.shape[0], -1])

class ChannelGate(nn.Layer):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum )
        scale = paddle.unsqueeze(scale, 2)
        scale = paddle.unsqueeze(scale, 3)
        scale = paddle.expand_as(scale, x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = paddle.reshape(tensor, shape=[tensor.shape[0], tensor.shape[1], -1])
    s, _ = paddle.max(tensor_flatten, axis=2, keepdim=True)
    tmp = paddle.exp(tensor_flatten - s)
    tmp = paddle.sum(tmp, axis=2, keepdim=True)
    tmp = paddle.log(tmp)
    outputs = s + tmp
    return outputs

class ChannelPool(nn.Layer):
    def forward(self, x):
        x1 = paddle.unsqueeze(paddle.max(x,1), 1)
        x2 = paddle.unsqueeze(paddle.mean(x, 1), 1)
        return paddle.concat( [x1, x2], axis=1 )

class SpatialGate(nn.Layer):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Layer):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class CALayer(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2D(channel,
                      channel // reduction,
                      1,
                      padding=0,
                      bias_attr=True), nn.ReLU(),
            nn.Conv2D(channel // reduction,
                      channel,
                      1,
                      padding=0,
                      bias_attr=True), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Layer):
    def __init__(self,
                 n_feat,
                 kernel_size,
                 reduction=16,
                 bn=False,
                 act=nn.ReLU(),
                 res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2D(n_feat, n_feat, kernel_size, padding='SAME'))
            if bn: modules_body.append(nn.BatchNorm2D(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Layer):
    def __init__(self, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                n_feat, kernel_size, reduction, bn=False, act=nn.ReLU(), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2D(n_feat, n_feat, kernel_size, padding='SAME'))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows = input.shape[2]
    filter_rows = weight.shape[2]
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation, groups=groups)


class _ConvNd(nn.Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        if transposed:
            # self.weight = Parameter(torch.Tensor(
            #     in_channels, out_channels // groups, *kernel_size))
            self.weight = self.create_parameter([in_channels, out_channels // groups, *kernel_size], dtype='float32',
                                                  default_initializer=paddle.nn.initializer.Uniform(-stdv, stdv))
        else:
            # self.weight = Parameter(torch.Tensor(
            #     out_channels, in_channels // groups, *kernel_size))
            self.weight = self.create_parameter([out_channels, in_channels // groups, *kernel_size], dtype='float32',
                                                  default_initializer=paddle.nn.initializer.Uniform(-stdv, stdv))
        self.bias = self.create_parameter([out_channels], is_bias=True, attr=bias, dtype='float32',
                                            default_initializer=paddle.nn.initializer.Uniform(-stdv, stdv))

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)


def act(act_type, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU()
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2D(nc)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2D(nc)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.Pad2D(padding, mode='reflect')
    elif pad_type == 'replicate':
        layer = nn.Pad2D(padding, mode='replicate')
    else:
        raise NotImplementedError('padding layer [%s] is not implemented' % pad_type)
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Layer):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [%s]' % mode
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


class ResidualDenseBlock_5C(nn.Layer):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                 norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc + gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc + 2 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc + 3 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc + 4 * gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(paddle.concat((x, x1), 1))
        x3 = self.conv3(paddle.concat((x, x1, x2), 1))
        x4 = self.conv4(paddle.concat((x, x1, x2, x3), 1))
        x5 = self.conv5(paddle.concat((x, x1, x2, x3, x4), 1))

        return x5 * 0.2


class DMDB2(nn.Layer):
    """
    DeMoireing  Dense Block
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                 norm_type=None, act_type='leakyrelu', mode='CNA', delia=1):
        super(DMDB2, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
                                          norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
                                          norm_type, act_type, mode)

        self.deli = nn.Sequential(
            Conv2d(64, 64, 3, stride=1, dilation=delia),
            nn.LeakyReLU(0.2),
        )
        self.deli2 = nn.Sequential(
            Conv2d(64, 64, 3, stride=1),
            nn.LeakyReLU(0.2),
        )
        # self.sam1 = SAM(64,64,1)
        # self.sam2 = SAM(64,64,1)

    def forward(self, x):
        # att1 = self.sam1(x)
        # att2 = self.sam2(x)

        out = self.RDB1(x)
        out = out + x
        out2 = self.RDB2(out)
        deli1 = self.deli(x)
        deli2 = 0.2 * self.deli2(self.deli(x))
        out3 = deli1 + deli2
        # out3 = self.deli(x) + 0.2 * self.deli2(self.deli(x))
        return out2 * 0.2 + out3



class depth_to_space(nn.Layer):
    def __init__(self,scale_factor):
        super(depth_to_space, self).__init__()
        self.scale_factor=scale_factor
    def forward(self,tensor):
        num, ch, height, width = tensor.shape
        if ch % (self.scale_factor * self.scale_factor) != 0:
            raise ValueError('channel of tensor must be divisible by '
                         '(scale_factor * scale_factor).')

        new_ch = ch // (self.scale_factor * self.scale_factor)
        new_height = height * self.scale_factor
        new_width = width * self.scale_factor

        tensor = tensor.reshape(
            [num, self.scale_factor, self.scale_factor, new_ch, height, width]).clone()
        # new axis: [num, new_ch, height, scale_factor, width, scale_factor]
        tensor = tensor.transpose([0, 3, 4, 1, 5, 2])
        tensor = tensor.reshape([num, new_ch, new_height, new_width])
        return tensor

class space_to_depth(nn.Layer):
    def __init__(self,scale_factor):
        super(space_to_depth, self).__init__()
        self.scale_factor=scale_factor
    def forward(self,tensor):
        num, ch, height, width = tensor.shape
        if height % self.scale_factor != 0 or width % self.scale_factor != 0:
            raise ValueError('height and widht of tensor must be divisible by '
                         'scale_factor.')

        new_ch = ch * (self.scale_factor * self.scale_factor)
        new_height = height // self.scale_factor
        new_width = width // self.scale_factor

        tensor = tensor.reshape(
            [num, ch, new_height, self.scale_factor, new_width, self.scale_factor]).clone()
        # new axis: [num, scale_factor, scale_factor, ch, new_height, new_width]
        tensor = tensor.transpose([0, 3, 5, 1, 2, 4])
        tensor = tensor.reshape([num, new_ch, new_height, new_width])
        return tensor

class adaptive_implicit_trans(nn.Layer):
    def __init__(self):
        super(adaptive_implicit_trans, self).__init__()
        conv_shape = (1,1,64,64)

        self.it_weights = paddle.create_parameter (
            shape = (1,1,64,1),dtype='float32',
            default_initializer =nn.initializer.Constant(value=1.0))

        kernel = np.zeros(conv_shape)
        r1 = sqrt(1.0/8)
        r2 = sqrt(2.0/8)
        for i in range(8):
            _u = 2*i+1
            for j in range(8):
                _v = 2*j+1
                index = i*8+j
                for u in range(8):
                    for v in range(8):
                        index2 = u*8+v
                        t = cos(_u*u*pi/16)*cos(_v*v*pi/16)
                        t = t*r1 if u==0 else t*r2
                        t = t*r1 if v==0 else t*r2
                        kernel[0,0,index2,index] = t
        self.kernel = paddle.to_tensor(kernel, dtype = 'float32')

    def forward(self, inputs):

        k = (self.kernel*self.it_weights).reshape([64,64,-1,1])


        y = nn.functional.conv2d(inputs,
                        k,
                        padding = 'SAME',groups=inputs.shape[1]//64,
                        data_format='NCHW')
        
        return y


class ScaleLayer(nn.Layer):
    def __init__(self, s):
        super(ScaleLayer, self).__init__()
        self.s = s
        self.kernel =paddle.create_parameter(
            shape = (1,),dtype='float32',
            default_initializer=nn.initializer.Constant(s))

    def forward(self, inputs):
        return inputs*self.kernel

class conv_rl(nn.Layer):
    def __init__(self,inf,filters, kernel, padding='SAME', use_bias = True, dilation_rate=1, strides=(1,1)):

        super(conv_rl, self).__init__()
        self.filters =filters
        self.kernel =kernel
        self.dilation_rate=dilation_rate
        if dilation_rate == 0:
            self.convr1= nn.Conv2D(inf,filters,1,padding=padding)
        else:
            self.convr2= nn.Conv2D(inf,filters,kernel,padding=padding,
                dilation=dilation_rate,
                stride=strides)
        self.relu= nn.ReLU()

    def forward(self,x):
        if self.dilation_rate == 0:
            y =self.convr1(x)
        else:
            y = self.convr2(x)
        y= self.relu(y)
        return y

class conv(nn.Layer):
    def __init__(self,inf,filters, kernel, padding='SAME', use_bias = True, dilation_rate=1, strides=(1,1)):
        super(conv, self).__init__()
        self.conv =nn.Conv2D(inf,filters,kernel,padding=padding,
            dilation=dilation_rate, stride=strides)
    def forward(self,x):
        y = self.conv(x)
        return y
class pre_block(nn.Layer):
    def __init__(self,d_list,nFilters, enbale = True):
        super(pre_block, self).__init__()
        self.d_list= d_list
        self.nFilters=nFilters
        self.enable=True
        layer_list = []
        for i in range(len(self.d_list)):
            j=i+1
            conv_rl_layer = conv_rl(self.nFilters+j*self.nFilters,
                                        self.nFilters,
                                        3,
                                        dilation_rate=self.d_list[i])
            layer_list.append(conv_rl_layer )
        self.cr_layers = nn.LayerList(layer_list)
        self.conv1=conv(self.nFilters*7,64,3)
        self.ad=adaptive_implicit_trans()
        self.conv2=conv(self.nFilters,self.nFilters*2,1)
        self.scale=ScaleLayer(s=0.1)
        self.lamb=lambda x: x*0
    def forward(self,x):
        t = x.detach()
        for layer in self.cr_layers:
            _t = layer(t)
            t = paddle.concat([_t,t],axis=1)
        
        t = self.conv1(t)
        
        t = self.ad(t)
        t = self.conv2(t)
        t = self.scale(t)
        if not self.enable:
            t = self.lamb(t)
        t =x+t
        return t

class pos_block(nn.Layer):
    def __init__(self,d_list,nFilters):
        super(pos_block, self).__init__()
        self.d_list= d_list
        self.nFilters=nFilters
        self.enable=True
        layer_list = []
        for i in range(len(self.d_list)):
            j=i+1
            conv_rl_layer = conv_rl(self.nFilters+j*self.nFilters,
                                        self.nFilters,
                                        3,
                                        dilation_rate=self.d_list[i])
            layer_list.append(conv_rl_layer )
        self.cr_layers1 = nn.LayerList(layer_list)
        self.cr2= conv_rl(self.nFilters*7, self.nFilters*2,1)

    def forward(self,x):
        t = x
        for layer in self.cr_layers1:
            _t = layer(t)
            t = paddle.concat([_t,t],axis=1)
        t = self.cr2(t)
        return t

class global_block(nn.Layer):
    def __init__(self,nFilters):
        super(global_block, self).__init__()
        self.nFilters=nFilters
        self.pad=nn.Pad2D(padding=1)
        self.cr0=conv_rl(nFilters*2, nFilters*4, 3,strides=(2,2))
        self.avp=nn.AdaptiveAvgPool2D(1)
        self.lin1=nn.Linear(nFilters*4,nFilters*16)
        self.relu=nn.ReLU()
        self.lin2=nn.Linear(nFilters*16,nFilters*8)
        self.lin3=nn.Linear(nFilters*8,nFilters*4)
        self.cr1=conv_rl(nFilters*2, nFilters*4, 1)
        self.cr2=conv_rl(nFilters*4, nFilters*2, 1)
    def forward(self,x):
        t = self.pad(x)
        t = self.cr0(t)
        t = self.avp(t)
        t= paddle.squeeze(t, axis=[-2,-1])

        t = self.lin1(t)
        t =self.relu(t)
        t = self.lin2(t)
        t = self.relu(t)       
        t = self.lin3(t)
        _t = self.cr1(x)
        t =t.unsqueeze(-1).unsqueeze(-1)
        _t = paddle.multiply(_t,t)
        _t =self.cr2(_t)
        return _t


class MBCNN(nn.Layer):
    def __init__(self, nFilters, multi=True):
        super(MBCNN, self).__init__()
        d_list_a = (1,2,3,2,1)
        d_list_b = (1,2,3,2,1)
        d_list_c = (1,2,2,2,1)
        self.nFilters = nFilters
        self.multi = multi
        self.std=space_to_depth(scale_factor=2)

        self.t1_conv_rl= conv_rl(12,self.nFilters*2,3, padding='SAME') 

        self.t1_pre_block=pre_block(d_list_a,self.nFilters, True)
        self.pad= nn.Pad2D(padding=1)
        self.t2_conv_rl=conv_rl(self.nFilters*2,self.nFilters*2,3, padding='VALID',strides=(2,2))  
        self.t2_pre_block=pre_block(d_list_b,self.nFilters,True)
        
        self.t3_conv_rl=conv_rl(self.nFilters*2,self.nFilters*2,3, padding='VALID',strides=(2,2)) 
        self.t3_pre_block=  pre_block(d_list_c, self.nFilters,True)
        self.t3_global_block=global_block(self.nFilters)
        self.t3_pos_block=pos_block(d_list_c,self.nFilters)
        self.t3_out_conv=conv(self.nFilters*2,12, 3)
        self.dts1=depth_to_space(scale_factor=2)
   

        self._t2_conv_rl=conv_rl(self.nFilters*2+3,self.nFilters*2, 1)
        self._t2_global_block1=global_block(self.nFilters)
        self._t2_pre_block=pre_block( d_list_b,self.nFilters,True)
        self._t2_global_block2=global_block(self.nFilters)
        self._t2_pos_block=pos_block(d_list_b,self.nFilters)

        self._t2_conv=conv(self.nFilters*2,12, 3)
        self.dts2=depth_to_space(scale_factor=2)

        self._t1_conv_rl=conv_rl(self.nFilters*2+3,self.nFilters*2, 1)
        self._t1_global_block = global_block(self.nFilters)
        self._t1_pre_block = pre_block( d_list_a, self.nFilters,True)
        self._t1_global_block = global_block(self.nFilters)
        self._t1_pos_block = pos_block( d_list_a,self.nFilters)
        self._t1_conv = conv(self.nFilters*2,12,3)
        self.dts3=depth_to_space(scale_factor=2)

    def forward(self,x):
        output_list = []
        _x = self.std(x)
        t1 = self.t1_conv_rl(_x)       #8m*8m
        t1 = self.t1_pre_block(t1)


        t2 = self.pad(t1)
        t2 = self.t2_conv_rl(t2)               #4m*4m
        t2 = self.t2_pre_block(t2)

        t3 = self.pad(t2)
        t3 = self.t3_conv_rl(t3)              #2m*2m
        t3 = self.t3_pre_block(t3)
        t3 = self.t3_global_block(t3)
        t3 = self.t3_pos_block(t3)   
        t3_out = self.t3_out_conv(t3)
        
        t3_out = self.dts1(t3_out)          #4m*4m
        output_list.append(t3_out)

        _t2 = paddle.concat([t3_out,t2],axis=1)
        _t2 = self._t2_conv_rl(_t2)
        _t2 = self._t2_global_block1(_t2)
        _t2 = self._t2_pre_block(_t2)
        _t2 = self._t2_global_block2(_t2)
        _t2 = self._t2_pos_block(_t2)
        t2_out = self._t2_conv(_t2)
        
        t2_out = self.dts2(t2_out)          #8m*8m

        output_list.append(t2_out)

        _t1 = paddle.concat([t1, t2_out],axis=1)
        _t1 = self._t1_conv_rl(_t1)
        _t1 = self._t1_global_block(_t1)
        _t1 = self._t1_pre_block(_t1)
        _t1 = self._t1_global_block(_t1)
        _t1 = self._t1_pos_block(_t1)
        _t1 = self._t1_conv(_t1)

        y = self.dts3(_t1)                        #16m*16m
        
        output_list.append(y)
        if self.multi != True:
            return y
        else:
            return output_list





class MBCNN_RCAN(nn.Layer):
    def __init__(self, nFilters, multi=True):
        super(MBCNN_RCAN, self).__init__()
        d_list_a = (1,2,3,2,1)
        d_list_b = (1,2,3,2,1)
        d_list_c = (1,2,2,2,1)
        self.nFilters = nFilters
        self.multi = multi
        self.std=space_to_depth(scale_factor=2)

        self.t1_conv_rl= conv_rl(12,self.nFilters*2,3, padding='SAME') 

        self.t1_pre_block=pre_block(d_list_a,self.nFilters, True)
        self.pad= nn.Pad2D(padding=1)
        self.t2_conv_rl=conv_rl(self.nFilters*2,self.nFilters*2,3, padding='VALID',strides=(2,2))  
        self.t2_pre_block=pre_block(d_list_b,self.nFilters,True)
        
        self.t3_conv_rl=conv_rl(self.nFilters*2,self.nFilters*2,3, padding='VALID',strides=(2,2)) 
        self.t3_pre_block=  pre_block(d_list_c, self.nFilters,True)
        self.t3_global_block=global_block(self.nFilters)
        self.t3_pos_block=pos_block(d_list_c,self.nFilters)
        self.t3_rcab = ResidualGroup(n_feat=nFilters*2, kernel_size=3, reduction=16, n_resblocks=10)
        self.t3_out_conv=conv(self.nFilters*2,12, 3)
        self.dts1=depth_to_space(scale_factor=2)
   

        self._t2_conv_rl=conv_rl(self.nFilters*2+3,self.nFilters*2, 1)
        self._t2_global_block1=global_block(self.nFilters)
        self._t2_pre_block=pre_block( d_list_b,self.nFilters,True)
        self._t2_global_block2=global_block(self.nFilters)
        self._t2_pos_block=pos_block(d_list_b,self.nFilters)
        self._t2_rcab = ResidualGroup(n_feat=nFilters*2, kernel_size=3, reduction=16, n_resblocks=10)
        self._t2_conv=conv(self.nFilters*2,12, 3)
        self.dts2=depth_to_space(scale_factor=2)

        self._t1_conv_rl=conv_rl(self.nFilters*2+3,self.nFilters*2, 1)
        self._t1_global_block = global_block(self.nFilters)
        self._t1_pre_block = pre_block( d_list_a, self.nFilters,True)
        self._t1_global_block = global_block(self.nFilters)
        self._t1_pos_block = pos_block( d_list_a,self.nFilters)
        self._t1_rcab = ResidualGroup(n_feat=nFilters*2, kernel_size=3, reduction=16, n_resblocks=10)
        self._t1_conv = conv(self.nFilters*2,12,3)
        self.dts3=depth_to_space(scale_factor=2)

    def forward(self,x):
        output_list = []
        _x = self.std(x)
        t1 = self.t1_conv_rl(_x)       #8m*8m
        t1 = self.t1_pre_block(t1)


        t2 = self.pad(t1)
        t2 = self.t2_conv_rl(t2)               #4m*4m
        t2 = self.t2_pre_block(t2)

        t3 = self.pad(t2)
        t3 = self.t3_conv_rl(t3)              #2m*2m
        t3 = self.t3_pre_block(t3)
        t3 = self.t3_global_block(t3)
        t3 = self.t3_pos_block(t3)
        t3 = self.t3_rcab(t3)           #rcab
        t3_out = self.t3_out_conv(t3)
        
        t3_out = self.dts1(t3_out)          #4m*4m
        output_list.append(t3_out)

        _t2 = paddle.concat([t3_out,t2],axis=1)
        _t2 = self._t2_conv_rl(_t2)
        _t2 = self._t2_global_block1(_t2)
        _t2 = self._t2_pre_block(_t2)
        _t2 = self._t2_global_block2(_t2)
        _t2 = self._t2_pos_block(_t2)
        _t2 = self._t2_rcab(_t2)
        t2_out = self._t2_conv(_t2)
        
        t2_out = self.dts2(t2_out)          #8m*8m

        output_list.append(t2_out)

        _t1 = paddle.concat([t1, t2_out],axis=1)
        _t1 = self._t1_conv_rl(_t1)
        _t1 = self._t1_global_block(_t1)
        _t1 = self._t1_pre_block(_t1)
        _t1 = self._t1_global_block(_t1)
        _t1 = self._t1_pos_block(_t1)
        _t1 = self._t1_rcab(_t1)
        _t1 = self._t1_conv(_t1)

        y = self.dts3(_t1)                        #16m*16m
        
        output_list.append(y)
        if self.multi != True:
            return y
        else:
            return output_list



class NonLocal(nn.Layer):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(NonLocal,self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2D(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2D(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Sequential(nn.Conv2D(in_channels = in_dim , out_channels = in_dim , kernel_size= 3, padding=1 ),
                                        nn.ReLU(),
                                        nn.Conv2D(in_channels = in_dim , out_channels = in_dim , kernel_size= 3, padding=1 ))            

        self.softmax  = nn.Softmax(axis=-1) #

        self.neighborhood_size=3
        self.relu=nn.ReLU()

        self.conv_out = nn.Conv2D(in_channels = in_dim*2 , out_channels = in_dim , kernel_size= 3, padding=1 )

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width ,height = x.shape
        proj_query  = self.query_conv(x).reshape([m_batchsize,-1, width*height]).transpose([0,2,1]) # B X C X (N)
        proj_key =  self.key_conv(x).reshape([m_batchsize,-1,width*height]) # B X C x (*W*H)

        energy =  paddle.bmm(proj_query,proj_key) # transpose check
        
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).reshape([m_batchsize,-1,width*height]) # B X C X N

        out = paddle.bmm(proj_value,attention.transpose([0,2,1]))
        out = out.reshape([m_batchsize,C,width,height])


        return self.conv_out(paddle.concat((out,x),axis=1))
      

class MBCNN_NL(nn.Layer):
    def __init__(self, nFilters, multi=True):
        super(MBCNN_NL, self).__init__()
        d_list_a = (1,2,3,2,1)
        d_list_b = (1,2,3,2,1)
        d_list_c = (1,2,2,2,1)
        self.nFilters = nFilters
        self.multi = multi
        self.std=space_to_depth(scale_factor=2)

        self.t1_conv_rl= conv_rl(12,self.nFilters*2,3, padding='SAME') 

        self.t1_pre_block=pre_block(d_list_a,self.nFilters, True)
        self.pad= nn.Pad2D(padding=1)
        self.t2_conv_rl=conv_rl(self.nFilters*2,self.nFilters*2,3, padding='VALID',strides=(2,2))  
        self.t2_pre_block=pre_block(d_list_b,self.nFilters,True)
        
        self.t3_conv_rl=conv_rl(self.nFilters*2,self.nFilters*2,3, padding='VALID',strides=(2,2)) 
        self.t3_pre_block=  pre_block(d_list_c, self.nFilters,True)
        self.t3_global_block=global_block(self.nFilters)
        self.t3_pos_block=pos_block(d_list_c,self.nFilters)
        self.t3_non_local = NonLocal(self.nFilters*2)
        self.t3_out_conv=conv(self.nFilters*2,12, 3)
        self.dts1=depth_to_space(scale_factor=2)
   

        self._t2_conv_rl=conv_rl(self.nFilters*2+3,self.nFilters*2, 1)
        self._t2_global_block1=global_block(self.nFilters)
        self._t2_pre_block=pre_block( d_list_b,self.nFilters,True)
        self._t2_global_block2=global_block(self.nFilters)
        self._t2_pos_block=pos_block(d_list_b,self.nFilters)
        self._t2_non_local = NonLocal(self.nFilters*2)
        self._t2_conv=conv(self.nFilters*2,12, 3)
        self.dts2=depth_to_space(scale_factor=2)

        self._t1_conv_rl=conv_rl(self.nFilters*2+3,self.nFilters*2, 1)
        self._t1_global_block = global_block(self.nFilters)
        self._t1_pre_block = pre_block( d_list_a, self.nFilters,True)
        self._t1_global_block = global_block(self.nFilters)
        self._t1_pos_block = pos_block( d_list_a,self.nFilters)
        self._t1_conv = conv(self.nFilters*2,12,3)
        self.dts3=depth_to_space(scale_factor=2)

    def forward(self,x):
        output_list = []
        _x = self.std(x)
        t1 = self.t1_conv_rl(_x)       #8m*8m
        t1 = self.t1_pre_block(t1)


        t2 = self.pad(t1)
        t2 = self.t2_conv_rl(t2)               #4m*4m
        t2 = self.t2_pre_block(t2)

        t3 = self.pad(t2)
        t3 = self.t3_conv_rl(t3)              #2m*2m
        t3 = self.t3_pre_block(t3)
        t3 = self.t3_global_block(t3)
        t3 = self.t3_pos_block(t3)   
        t3 = self.t3_non_local(t3)
        t3_out = self.t3_out_conv(t3)
        t3_out = self.dts1(t3_out)          #4m*4m
        output_list.append(t3_out)

        _t2 = paddle.concat([t3_out,t2],axis=1)
        _t2 = self._t2_conv_rl(_t2)
        _t2 = self._t2_global_block1(_t2)
        _t2 = self._t2_pre_block(_t2)
        _t2 = self._t2_global_block2(_t2)
        _t2 = self._t2_pos_block(_t2)
        # _t2 = self._t2_non_local(_t2)
        t2_out = self._t2_conv(_t2)
        
        t2_out = self.dts2(t2_out)          #8m*8m

        output_list.append(t2_out)

        _t1 = paddle.concat([t1, t2_out],axis=1)
        _t1 = self._t1_conv_rl(_t1)
        _t1 = self._t1_global_block(_t1)
        _t1 = self._t1_pre_block(_t1)
        _t1 = self._t1_global_block(_t1)
        _t1 = self._t1_pos_block(_t1)
        _t1 = self._t1_conv(_t1)

        y = self.dts3(_t1)                        #16m*16m
        
        output_list.append(y)
        if self.multi != True:
            return y
        else:
            return output_list


class MBCNN_CBAM(nn.Layer):
    def __init__(self, nFilters, multi=True):
        super(MBCNN_CBAM, self).__init__()
        d_list_a = (1,2,3,2,1)
        d_list_b = (1,2,3,2,1)
        d_list_c = (1,2,2,2,1)
        self.nFilters = nFilters
        self.multi = multi
        self.std=space_to_depth(scale_factor=2)

        self.t1_conv_rl= conv_rl(12,self.nFilters*2,3, padding='SAME') 

        self.t1_pre_block=pre_block(d_list_a,self.nFilters, True)
        self.pad= nn.Pad2D(padding=1)
        self.t2_conv_rl=conv_rl(self.nFilters*2,self.nFilters*2,3, padding='VALID',strides=(2,2))  
        self.t2_pre_block=pre_block(d_list_b,self.nFilters,True)
        
        self.t3_conv_rl=conv_rl(self.nFilters*2,self.nFilters*2,3, padding='VALID',strides=(2,2)) 
        self.t3_pre_block=  pre_block(d_list_c, self.nFilters,True)
        self.t3_global_block=global_block(self.nFilters)
        self.t3_pos_block=pos_block(d_list_c,self.nFilters)
        self.t3_cbam = CBAM(nFilters*2)
        self.t3_out_conv=conv(self.nFilters*2,12, 3)
        self.dts1=depth_to_space(scale_factor=2)
   

        self._t2_conv_rl=conv_rl(self.nFilters*2+3,self.nFilters*2, 1)
        self._t2_global_block1=global_block(self.nFilters)
        self._t2_pre_block=pre_block( d_list_b,self.nFilters,True)
        self._t2_global_block2=global_block(self.nFilters)
        self._t2_pos_block=pos_block(d_list_b,self.nFilters)
        self._t2_cbam = CBAM(nFilters*2)
        self._t2_conv=conv(self.nFilters*2,12, 3)
        self.dts2=depth_to_space(scale_factor=2)

        self._t1_conv_rl=conv_rl(self.nFilters*2+3,self.nFilters*2, 1)
        self._t1_global_block = global_block(self.nFilters)
        self._t1_pre_block = pre_block( d_list_a, self.nFilters,True)
        self._t1_global_block = global_block(self.nFilters)
        self._t1_pos_block = pos_block( d_list_a,self.nFilters)
        self._t1_cbam = CBAM(nFilters*2)
        self._t1_conv = conv(self.nFilters*2,12,3)
        self.dts3=depth_to_space(scale_factor=2)

    def forward(self,x):
        output_list = []
        _x = self.std(x)
        t1 = self.t1_conv_rl(_x)       #8m*8m
        t1 = self.t1_pre_block(t1)


        t2 = self.pad(t1)
        t2 = self.t2_conv_rl(t2)               #4m*4m
        t2 = self.t2_pre_block(t2)

        t3 = self.pad(t2)
        t3 = self.t3_conv_rl(t3)              #2m*2m
        t3 = self.t3_pre_block(t3)
        t3 = self.t3_global_block(t3)
        t3 = self.t3_pos_block(t3)
        t3 = self.t3_cbam(t3)           #rcab
        t3_out = self.t3_out_conv(t3)
        
        t3_out = self.dts1(t3_out)          #4m*4m
        output_list.append(t3_out)

        _t2 = paddle.concat([t3_out,t2],axis=1)
        _t2 = self._t2_conv_rl(_t2)
        _t2 = self._t2_global_block1(_t2)
        _t2 = self._t2_pre_block(_t2)
        _t2 = self._t2_global_block2(_t2)
        _t2 = self._t2_pos_block(_t2)
        _t2 = self._t2_cbam(_t2)
        t2_out = self._t2_conv(_t2)
        
        t2_out = self.dts2(t2_out)          #8m*8m

        output_list.append(t2_out)

        _t1 = paddle.concat([t1, t2_out],axis=1)
        _t1 = self._t1_conv_rl(_t1)
        _t1 = self._t1_global_block(_t1)
        _t1 = self._t1_pre_block(_t1)
        _t1 = self._t1_global_block(_t1)
        _t1 = self._t1_pos_block(_t1)
        _t1 = self._t1_cbam(_t1)
        _t1 = self._t1_conv(_t1)

        y = self.dts3(_t1)                        #16m*16m
        
        output_list.append(y)
        if self.multi != True:
            return y
        else:
            return output_list


if __name__ == '__main__':
    
    net = MBCNN_NL(64)
    # net = NonLocal(128)
    # net = ResidualGroup(128,3,16,20)
    net = CBAM(128)
    img = paddle.randn([1,128,64,64])
    with paddle.no_grad():
        out = net(img)
        print(out.shape)
        # for i in out:
        #     print(i.shape)
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # 获取参数情况
    for p in net.parameters():
        mulValue = np.prod(p.shape)  # 使用numpy prod接口计算数组所有元素之积
        Total_params += mulValue  # 总参数量
        if p.stop_gradient:
            NonTrainable_params += mulValue  # 可训练参数量
        else:
            Trainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
