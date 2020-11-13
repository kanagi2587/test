import ResNet
import AP3D
import inflate
import getresnet
import oneflow as flow
import numpy as np
import oneflow.typing as tp
from typing import Tuple
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
def test_resnet(
    inputs:tp.Numpy.Placeholder((64, 3, 26, 110, 110))
) -> Tuple[tp.Numpy, tp.Numpy]:
    resnet2d=getresnet.getResnet()
    layer=resnet2d[0]
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[1, 3],[1, 3, 5],[]]
    y,f=ResNet.ResNet503D(10, AP3D.APP3DC, c3d_idx, nl_idx).build_network(inputs)
 
    return (y, f)



inputs=np.random.randint(-10,10,(64, 3, 26, 110, 110)).astype(np.float32)
#out=test_job(inputs)
y,f=test_resnet(inputs)
# print(y)
# print(f)