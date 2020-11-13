import math
import copy
import oneflow as flow
import oneflow.nn as nn
import inflate
import AP3D
import NonLocal
import getresnet
__all__ = ['AP3DResNet50', 'AP3DNLResNet50']

global time
time=0
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
class Bottleneck3D(object):
    def __init__(self, bottleneck2d, block, inflate_time=False, temperature=4, contrastive_att=True):
        super(Bottleneck3D, self).__init__()
        self.inflate_time=inflate_time
        self.bottleneck2d=bottleneck2d
        self.block=block
        self.temperature=temperature
        self.contrastive_att=contrastive_att
    def _inflate_downsample(self,inputs ,downsample2d, time_stride=1):
        global time 
        out=inflate.inflate_conv(
            inputs,downsample2d.conv2d,time_dim=1,time_stride=time_stride,times=time)
        time+=1
        #out=inflate.inflate_batch_norm(inputs,downsample2d[1]))
        return out
    def build_network(self,inputs):
        global time
        print("inputs:",inputs.shape)
        residual = inputs
        out=inflate.inflate_conv(
            inputs,
            self.bottleneck2d.conv1,
            time_dim=1,
            times=time)
        print("out1:",out.shape)

        time+=1
        # out=inflate.inflate_batch_norm(
        #     out,
        #     bottleneck2d.bn1
        # )

        out=nn.relu(out)
        if self.inflate_time == True:
            out=self.block(self.bottleneck2d.conv2, temperature=self.temperature,
                    contrastive_att=self.contrastive_att).build_network(out)
        else:
            out=inflate.inflate_conv(out,self.bottleneck2d.conv2, time_dim=1, times=time)
        time+=1
        print("out2:",out.shape)

       # out=inflate.inflate_batch_norm(out,bottleneck2d.bn2)
        out=nn.relu(out)

        out=inflate.inflate_conv(out,self.bottleneck2d.conv3, time_dim=1, times=time)
        time+=1
        print("out3:",out.shape)

       # out=inflate.inflate_batch_norm(out,bottleneck2d.bn3)

        if self.bottleneck2d.downsample is not None:
            residual=self._inflate_downsample(inputs,self.bottleneck2d.downsample)
        print("out,",out.shape)
        print("residual",residual.shape)
        out+=residual
          #  residual=inflate_batch_norm(residual,downsample2d[1]))
        out=nn.relu(out)

        return out

class ResNet503D(object):
    def __init__(self, num_classes, block, c3d_idx, nl_idx, temperature=4, contrastive_att=True, **kwargs):
        super(ResNet503D, self).__init__()
        self.block = block
        self.temperature = temperature
        self.contrastive_att = contrastive_att
        self.num_classes=num_classes
        self.c3d_idx=c3d_idx
        self.nl_idx=nl_idx

    
    def _inflate_reslayer(self,inputs,reslayer2d, c3d_idx, nonlocal_idx=[], nonlocal_channels=0):
        for i,layer2d in enumerate(reslayer2d):
            if i not in c3d_idx:
                layer3d = Bottleneck3D(layer2d, AP3D.C2D, inflate_time=False).build_network(inputs)
            else:
                layer3d = Bottleneck3D(layer2d, self.block, inflate_time=True, \
                                       temperature=self.temperature, contrastive_att=self.contrastive_att).build_network(inputs)
            inputs=layer3d

            if i in nonlocal_idx:
                layer3d = NonLocal.NonLocalBlock3D(nonlocal_channels, sub_sample=True).build_network(inputs)
                
        return layer3d

    def build_network(self,inputs):
        # resnet2d = torchvision.models.resnet50(pretrained=True)
        global time
        resnet2d=getresnet.getResnet()
        conv2d=Conv2d(3,64,[7,7],[2,2],[3,3],dilation=[1, 1])

        out= inflate.inflate_conv(inputs,conv2d, time_dim=1,times=time) 

        time+=1       
     #   inputs=inflate.inflate_batch_norm(inputs,resnet2d.bn1)
        out=nn.relu(out)
        out=inflate.inflate_pool(out,kernel_size=3,padding=1,stride=2,dilation=1, time_dim=1)
        out=self._inflate_reslayer(out,resnet2d[0], c3d_idx=self.c3d_idx[0], 
                                             nonlocal_idx=self.nl_idx[0], nonlocal_channels=256)
        print("layer1finish")
        out=self._inflate_reslayer(out,resnet2d[1], c3d_idx=self.c3d_idx[1], \
                                             nonlocal_idx=self.nl_idx[1], nonlocal_channels=512)
        print("layer2finish")

        out=self._inflate_reslayer(out,resnet2d[2], c3d_idx=self.c3d_idx[2], \
                                             nonlocal_idx=self.nl_idx[2], nonlocal_channels=1024)
        print("layer3finish")
        
        out= self._inflate_reslayer(out,resnet2d[3], c3d_idx=self.c3d_idx[3], \
                                             nonlocal_idx=self.nl_idx[3], nonlocal_channels=2048)
        print("layer4finish")

        b,c,t,h,w=out.shape
        out =flow.transpose(out,perm=[0,2,1,3,4])
        out=flow.reshape(out,shape=[b*t, c, h, w])
        out=nn.max_pool2d(
            input=out,
            ksize=out.shape[2:],
            strides=None,
            padding="VALID"
        )
        out=flow.reshape(out,shape=[b,t,-1])
        out=flow.math.reduce_mean(out,axis=1)

        f = flow.layers.batch_normalization(inputs=out,
                                                 momentum=0.997,
                                                 epsilon=1.001e-5,
                                                 center=True,
                                                 scale=True,
                                                 trainable=True,
                                                 name= "Resnet503D_linear_bn"+str(time))
        time+=1
        kernel_initializer = flow.variance_scaling_initializer(2, 'fan_in', 'random_normal')
        weight_regularizer = flow.regularizers.l2(1.0 / 32768) 
        y=flow.layers.dense(        out, 
                                   self.num_classes, 
                                   use_bias=True,
                                   bias_initializer=kernel_initializer,
                                   kernel_regularizer=weight_regularizer,
                                   bias_regularizer=weight_regularizer,                                   
                                   trainable=True)
        return y,f
def AP3DResNet50(num_classes, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[],[],[]]
    return ResNet503D(num_classes, AP3D.APP3DC, c3d_idx, nl_idx, **kwargs)

def AP3DNLResNet50(num_classes, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[1, 3],[1, 3, 5],[]]
    return ResNet503D(num_classes, AP3D.APP3DC, c3d_idx, nl_idx, **kwargs)
