import oneflow as flow
import oneflow.nn as nn
import numpy as np
import datetime
def _get_kernel_initializer():
    return flow.variance_scaling_initializer(distribution="random_normal", data_format="NCHW")

def _get_regularizer():
    return flow.regularizers.l2(0.00005)

def _get_bias_initializer():
    return flow.zeros_initializer()
def conv3d_layer(
    name,
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    padding="VALID",
    data_format="NCDHW",
    dilation_rate=1,
    activation="Relu",
    use_bias=False,
    groups=1,
    weight_initializer=_get_kernel_initializer(),
    bias_initializer=_get_bias_initializer(),
    weight_regularizer=_get_regularizer(),
    bias_regularizer=_get_regularizer(),
    trainable=True
):
    if isinstance(kernel_size,int):
        kernel_size_1=kernel_size
        kernel_size_2 = kernel_size
        kernel_size_3 = kernel_size
    if isinstance(kernel_size,list):
        kernel_size_1=kernel_size[0]
        kernel_size_2=kernel_size[1]
        kernel_size_3=kernel_size[2]

    weight_shape=(filters,inputs.shape[1]//groups,kernel_size_1,kernel_size_2,kernel_size_3)
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=inputs.dtype,
        initializer=weight_initializer,
        regularizer=weight_regularizer,
        trainable=trainable
    )
    output=flow.nn.conv3d(
         inputs, weight, strides, padding, data_format, dilation_rate, groups, name=name
    )
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=inputs.dtype,
            initializer=bias_initializer,
            regularizer=bias_regularizer,
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == "Relu":
            output = flow.nn.relu(output)
        else:
            raise NotImplementedError

    return output

def conv3d_layer_with_bn(
    name,
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    padding="VALID",
    data_format="NCDHW",
    dilation_rate=1,
    activation="Relu",
    use_bias=False,
    groups=1,
    weight_initializer=_get_kernel_initializer(),
    bias_initializer=_get_bias_initializer(),
    weight_regularizer=_get_regularizer(),
    bias_regularizer=_get_regularizer(),
    use_bn=True,
):
    output = conv3d_layer(name=name,
                          inputs=inputs,
                          filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          data_format=data_format,
                          dilation_rate=dilation_rate,
                          activation=activation,
                          use_bias=use_bias,
                          groups=groups,
                          weight_initializer=weight_initializer,
                          bias_initializer=bias_initializer,
                          weight_regularizer=weight_regularizer,
                          bias_regularizer=bias_regularizer)

    if use_bn:
        output = flow.layers.batch_normalization(inputs=output,
                                                 axis=1,
                                                 momentum=0.997,
                                                 epsilon=1.001e-5,
                                                 center=True,
                                                 scale=True,
                                                 trainable=True,
                                                 name=name + "_bn")
    return output


def inflate_conv(inputs,
                conv2d,
                time_dim=1,
                time_padding=0,
                time_stride=1,
                time_dilation=1,
                center=False,
                times=0):
    name=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    kernel_dim=[time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1]]
    if isinstance(conv2d.padding,int):
        padding1=conv2d.padding
        padding2=conv2d.padding
    if isinstance(conv2d.padding,list):
        padding1=conv2d.padding[0]
        padding2=conv2d.padding[1]
    padding = [0,0,time_padding, padding1, padding2]
    stride = [time_stride, conv2d.stride[0], conv2d.stride[0]]
    if isinstance(conv2d.dilation,int):
        dilation1=conv2d.dilation
        dilation2 = conv2d.dilation
    if isinstance(conv2d.dilation,list):
        dilation1=conv2d.dilation[0]
        dilation2=conv2d.dilation[1]
    dilation = [time_dilation, dilation1, dilation2]

    # weight_2d=conv2d.weight.data
    # if center:
    #     weight_3d=np.zeros(weight_2d.shape)
    #     weight_3d=np.expand_dims(weight_3d,axis=2)
    #     weight_3d=np.tile(weight_3d,(1,1,time_dim,1,1))
    #     middle_idx = time_dim // 2
    #     weight_3d[:, :, middle_idx, :, :] = weight_2d
    # else:
    #     weight_3d=np.expand_dims(weight_3d,axis=2)
    #     weight_3d=np.tile(weight_3d,(1,1,time_dim,1,1))
    #     weight_3d=weight_3d/time_dim
    
    # init=flow.constant_initializer(weight_3d)
    init=flow.random_uniform_initializer(minval=0, maxval=1)

    output=conv3d_layer(
        "inflate_conv_"+str(times)+"_"+name,inputs,conv2d.out_channels,kernel_size=kernel_dim,
        dilation_rate=dilation,strides=stride,
        padding=padding,
        weight_initializer=init
    )
    return output
def inflate_linear(inputs,linear2d,time_dim):
    # weight3d=linear2d.weight.data
    # weight3d=np.tile(weight3d,(1,time_dim))
    # weight3d=weight3d/time_dim
    # init=flow.constant_initializer(weight3d)
    init=flow.random_uniform_initializer(minval=0, maxval=0.5)
    kernel_initializer = flow.variance_scaling_initializer(2, 'fan_in', 'random_normal')
    weight_regularizer = flow.regularizers.l2(1.0 / 32768) 
    linear3d = flow.layers.dense(   inputs, 
                                   linear2d.out_features, 
                                   use_bias=True,
                                   bias_initializer=kernel_initializer,
                                   kernel_regularizer=weight_regularizer,
                                   bias_regularizer=weight_regularizer,
                                   trainable=True)
    return linear3d

def inflate_batch_norm(inputs,batch2d):
    output = flow.layers.batch_normalization(inputs=output,
                                            axis=batch2d.num_features,
                                            momentum=0.997,
                                            epsilon=1.001e-5,
                                            center=True,
                                                 scale=True,
                                                 trainable=True,
                                                 name=name + "_bn")
    return output
            


def inflate_pool(inputs,
                kernel_size,
                padding,
                stride,
                dilation,
                avg_type=False,
                time_dim=1,
                time_padding=0,
                time_stride=None,
                time_dilation=1):
    kernel_dim = [time_dim, kernel_size, kernel_size]
    padding = [0,0,time_padding, padding, padding]
    if time_stride is None:
        time_stride=time_dim
    stride = [time_stride, stride, stride]
    if avg_type==False:
        dilation = [time_dilation, dilation,dilation]
        pool3d=nn.max_pool3d(
            inputs,
            ksize=kernel_dim,
            strides=stride,
            padding=padding
        )
    else:
        pool3d=nn.avg_pool3d(
            inputs,
            ksize=kernel_dim,
            strides=stride,
            padding=padding
        )
    return pool3d