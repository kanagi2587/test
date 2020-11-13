import oneflow as flow
import numpy as np
import oneflow.typing as tp
import AP3D
import ResNet
import torch
import getresnet
import inflate
class Conv2d(object):
    def __init__(self, in_channels, out_channels, kernel_size,stride=1,padding=1,bias=False,dilation=1):
        super(Conv2d, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.bias=bias
        self.dilation=dilation
@flow.global_function()
def test_job(
    inputs:tp.Numpy.Placeholder((64, 256, 26, 28, 28))
)->tp.Numpy:
    resnet2d=getresnet.getResnet()
    layer=resnet2d[1]
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[1, 3],[1, 3, 5],[]]
    conv2d=Conv2d(3,64,[7,7],[2,2],[3,3],dilation=[1, 1])
    print(layer[0].conv2.kernel_size)
    print(layer[0].conv2.padding)
    print(layer[0].conv2.stride)
    out=inflate.inflate_conv(
            inputs,
            layer[0].conv1,
            time_dim=1,
            times=1)
    # output=_inflate_reslayer(inputs,layer, c3d_idx=c3d_idx[0], \
    #                                          nonlocal_idx=nl_idx[0], nonlocal_channels=256)
    #output=inflate.inflate_conv(inputs,1,layer[1].conv2, time_dim=1)
    #output=inflate.inflate_conv(output,2,layer[1].conv2, time_dim=1)
    # output=AP3D.C2D(layer[1].conv2).build_network(inputs,1)
    # output=AP3D.C2D(layer[1].conv2).build_network(output,2)
    return out
inputs=np.random.randint(-10,10,(64, 256, 26, 28, 28)).astype(np.float32)
out=test_job(inputs)
print(out.shape)

