import oneflow as flow
import oneflow.nn as nn
import numpy as np
import datetime

#def constantPad3d(padding,value):

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
    tempname=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')    

    if isinstance(kernel_size,int):
        kernel_size_1=kernel_size
        kernel_size_2 = kernel_size
        kernel_size_3 = kernel_size
    if isinstance(kernel_size,list):
        kernel_size_1=kernel_size[0]
        kernel_size_2=kernel_size[1]
        kernel_size_3=kernel_size[2]
    print(filters)
    print(inputs.shape[1])
    print(groups)
    weight_shape=(filters,inputs.shape[1]//groups,kernel_size_1,kernel_size_2,kernel_size_3)
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=inputs.dtype,
        initializer=weight_initializer,
        regularizer=weight_regularizer,
        trainable=trainable
    )
    print(weight_shape)
    print(weight.shape)
    output=flow.nn.conv3d(
         inputs, weight, strides, padding, data_format, dilation_rate, groups, name=name+tempname
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

def conv2d_layer(
    name,
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    padding="VALID",
    data_format="NCHW",
    dilation_rate=1,
    activation="Relu",
    use_bias=False,
    groups=1,
    weight_initializer=_get_kernel_initializer(),
    bias_initializer=_get_bias_initializer(),
    weight_regularizer=_get_regularizer(),
    bias_regularizer=_get_regularizer(),
):
    if isinstance(kernel_size, int):
        kernel_size_1 = kernel_size
        kernel_size_2 = kernel_size
    if isinstance(kernel_size, list):
        kernel_size_1 = kernel_size[0]
        kernel_size_2 = kernel_size[1]

    weight_shape = (filters, inputs.shape[1]//groups, kernel_size_1, kernel_size_2)

    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=inputs.dtype,
        initializer=weight_initializer,
        regularizer=weight_regularizer,
    )
    output = flow.nn.conv2d(
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
    data_format="NCHW",
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


def conv2d_layer_with_bn(
    name,
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    padding="VALID",
    data_format="NCHW",
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
    output = conv2d_layer(name=name,
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



class APM(object):
    def __init__(self, in_channels, out_channels, time_dim=3, temperature=4, contrastive_att=True):
        super(APM,self).__init__()
        self.time_dim=time_dim
        self.temperature=temperature
        self.contrastive_att=contrastive_att
        self.in_channels=in_channels
        self.out_channels=out_channels
        #padding=[0,0,0,0,(time_dim-1)//2,(time_dim-1)//2]
        #self.padding=nn.ConstantPad3d(padding,value=0)
    def build_network(self,inputs):
        b,c,t,h,w=inputs.shape
        N=self.time_dim
        templist=[np.arange(0,t)+i for i in range(N) if i!=N//2]
        templist=np.expand_dims(templist,axis=0)
        neighbor_time_index=np.concatenate(
            templist,axis=0
        )
        # neighbor_time_index=flow.concat(
        #     templist,axis=0
        # )
        neighbor_time_index=np.transpose(neighbor_time_index)
        neighbor_time_index=np.ndarray.flatten(neighbor_time_index)
        #寻找tensor.long的代替（把tensor变成longtensor）
        #tensor 中long 是64整形
        neighbor_time_index=np.int64(neighbor_time_index)

        semantic=conv3d_layer("conv_semantic_",inputs,self.out_channels,
            kernel_size=1,use_bias=False,padding="SAME"
        )

        inputs_norm=flow.math.l2_normalize(
            semantic,axis=1
        )


        inputs_norm_padding=flow.pad(inputs_norm,paddings=[
            (0,0),(0,0),((self.time_dim-1)//2,(self.time_dim-1)//2), (0,0),(0,0)]
        )
        inputs_norm_expand=flow.expand_dims(inputs_norm,axis=3)
        temp_inputs_norm_expand=inputs_norm_expand
        for i in range(N-2):
            inputs_norm_expand=flow.concat(
               inputs=[ inputs_norm_expand,temp_inputs_norm_expand],
                axis=3
            )
       
        #inputs_norm_expand=flow.transpose(inputs_norm_expand,perm=[0, 2, 3, 4, 5, 1])
        print("inputs_norm_expand",inputs_norm_expand.shape)
        inputs_norm_expand=flow.reshape(inputs_norm_expand,(inputs_norm_expand.shape[0],inputs_norm_expand.shape[2],inputs_norm_expand.shape[3],inputs_norm_expand.shape[4],inputs_norm_expand.shape[5],inputs_norm_expand.shape[1]))
        inputs_norm_expand=flow.reshape(inputs_norm_expand,shape=[-1, h*w, c//16])

        slice_list=[]
        for index in  neighbor_time_index:
            temp=flow.slice(
                inputs_norm_padding,
                begin=[None,None,int(index),None,None],
                #size=[None,slice_shape[1],1,slice_shape[3],slice_shape[4]]
                size=[None,None,1,None,None]
            )      
            slice_list.append(temp)
            neighbor_norm=flow.concat(
            slice_list,axis=2
        )
        neighbor_norm=flow.transpose(neighbor_norm,perm=[0, 2, 1, 3, 4])
        #inputs_norm_expand=flow.reshape(neighbor_norm,(neighbor_norm.shape[0],neighbor_norm.shape[2],neighbor_norm.shape[3],neighbor_norm.shape[4],neighbor_norm.shape[5],neighbor_norm.shape[1]))

        neighbor_norm=flow.reshape(neighbor_norm,shape=[-1, c//16, h*w])
        similarity=flow.matmul(inputs_norm_expand,neighbor_norm)*self.temperature
        similarity=nn.softmax(similarity,axis=-1)
        inputs_padding=flow.pad(inputs,
        paddings=[
            (0,0),(0,0),((self.time_dim-1)//2,(self.time_dim-1)//2), (0,0),(0,0)]
        ) 
        #neighbor=inputs_padding[:, :, neighbor_time_index, :, :]
        slice_list=[]
        for index in  neighbor_time_index:
            temp=flow.slice(
                inputs_padding,
                begin=[None,None,int(index),None,None],
                size=[None,None,1,None,None]
            )      
            slice_list.append(temp)
        neighbor=flow.concat(
            slice_list,axis=2
        )
        neighbor=flow.transpose(neighbor,perm=[0,2,3,4,1])

        neighbor=flow.reshape(neighbor,shape=[-1, h*w, c]) 

        neighbor_new=flow.matmul(similarity,neighbor)
        neighbor_new=flow.reshape(neighbor_new,shape=[b, t*(N-1), h, w, c])
        neighbor_new=flow.transpose(neighbor_new,perm=[0, 4, 1, 2, 3])
        if self.contrastive_att:
            
            temp_input=flow.expand_dims(inputs,axis=3)
            temp_temp_input=temp_input
            temp_input=flow.concat(
                inputs=[temp_input,temp_temp_input],axis=3
            )
            temp_input=flow.reshape(temp_input,shape=[b, c, (N-1)*t, h, w])
            input_att=conv3d_layer(
                "conv3d_inputmapping",temp_input,self.out_channels,
                kernel_size=1, use_bias=False,trainable=False
            )

            n_att=conv3d_layer(
                "conv3d_nmapping",neighbor_new,self.out_channels,
                kernel_size=1, use_bias=False,trainable=False
            )
            contrastive_att_net=conv3d_layer(
                "conv3d_att_net",input_att*n_att,self.out_channels,
                kernel_size=1, use_bias=False
            )
            constastive_att=flow.math.sigmoid(contrastive_att_net)
            neighbor_new = neighbor_new * self.contrastive_att

            #device 暂时先空着了
        input_offset=np.zeros([b, c, N*t, h, w],dtype=np.float)
        
        init = flow.zeros_initializer()
        input_offset = flow.get_variable(
        "input_offset",
        shape=(b, c, N*t, h, w),
        initializer=init,
        dtype=inputs.dtype,
        trainable=True
        )
        input_index=np.array(
            [i for i in range(t*N) if i%N==N//2]
        )
        neighbor_index=np.array(
            [i for i in range(t*N) if i%N!=N//2])
        # print("inputs: ",inputs.shape)
        # print("input_index:",input_index)
        # print("input_index_len:",len(input_index))
        print("input_offset:",input_offset.shape)
        input_offset_list=[]
        inputs_list=[]
        neighbor_new_list=[]
        for index in  range(input_offset.shape[2]):
            temp=flow.slice(
                input_offset,
                begin=[None,None,int(index),None,None],
                size=[None,None,1,None,None]
            )  
            input_offset_list.append(temp)
        for index in range(inputs.shape[2]):
            temp=flow.slice(
                inputs,
                begin=[None,None,int(index),None,None],
                size=[None,None,1,None,None]
            )
            inputs_list.append(temp)
        for index in range(neighbor_new.shape[2]):
            temp=flow.slice(
                neighbor_new,
                begin=[None,None,int(index),None,None],
                size=[None,None,1,None,None]
            )
            neighbor_new_list.append(temp)
        temp_index=0
        for index in input_index:
            input_offset_list[index]+=inputs_list[temp_index]
            temp_index+=1
        # print("neighbor_new:",neighbor_new.shape)
        # print("neighbor_index:",neighbor_index.shape)
        temp_index=0
        for index in neighbor_index:
            input_offset_list[index]+=neighbor_new_list[temp_index]
            temp_index+=1
        # print("before",input_offset.shape)
        input_offset=flow.concat(
            input_offset_list,axis=2
        )
        print("after",input_offset.shape)

        return input_offset


class C2D(object):
    def __init__(self, conv2d, **kwargs):
        super(C2D, self).__init__()
        self.conv2d=conv2d
        self.kernel_dim = [1, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride = [1, conv2d.stride[0], conv2d.stride[0]]
        self.padding = [0,0,0, conv2d.padding[0], conv2d.padding[1]]
        
    def build_network(self,inputs):
        # weight_2d=self.conv2d.weight.data
        # weigt_3d=np.zeros(weight_2d.shape)   
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d[:, :, 0, :, :] = weight_2d
        #init=flow.constant_initializer(weight_3d)
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)
        output=conv3d_layer("conv_c2d_",inputs=inputs,filters=self.conv2d.out_channels,
                kernel_size=self.kernel_dim,strides=self.stride, padding=self.padding,
                use_bias=True,weight_initializer=init
        )
        return output
class I3D(object):
    def _init_(self,conv2d,time_dim=3,time_stride=1,**kwargs):
        super(I3D,self)._init_()
        self.kernel_dim=[time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[time_stride, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,time_dim//2, conv2d.padding[0], conv2d.padding[1]]
        self.conv2d=conv2d
        self.time_dim=time_dim
    def build_network(self,inputs):
        #pytorch中的repeat ==>numpy tile
        #由于上面使用了numpy的zeros函数导致weight3d 变成了np类型的对象，无法使用
        #flow相关的函数，因此这里的后续补充需要从zero开始。
        # oneflow.repeat(input: oneflow.python.framework.remote_blob.BlobDef, repeat_num: int, 
        # name: Optional[str] = None) → oneflow.python.framework.remote_blob.BlobDef
        #weight_3d=flow.repeat(weight_3d,)
        # weight_2d=self.conv2d.weight.data
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d=np.tile(weight_3d,(1,1,self.time_dim,1,1))
        # middle_dix=self.time_dim//2
        # weight_3d[:, :, middle_idx, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)

        output=conv3d_layer("conv_I3D_",inputs,self.conv2d.out_channels,
                kernel_size=self.kernel_dim,strides=self.stride, padding=self.padding,
                use_bias=True, weight_initializer=init
        )
        return output
class API3D(object):
    def __init__(self, conv2d, time_dim=3, time_stride=1, temperature=4, contrastive_att=True):
        super(API3D, self).__init__()
        self.conv2d=conv2d
        self.time_dim=time_dim
        self.APM=APM(conv2d.in_channels, conv2d.in_channels//16, 
                       time_dim=time_dim, temperature=temperature, contrastive_att=contrastive_att)
        self.kernel_dim=[time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[time_stride*time_dim, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,0, conv2d.padding[0], conv2d.padding[1]]
    def build_network(self,inputs):
        # weight_2d = self.conv2d.weight.data
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d=np.tile(
        #     weight_3d,(1,1,self.time_dim,1,1)
        # )
        # middle_idx = self.time_dim // 2
        # weight_3d[:, :, middle_idx, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)

        init=flow.random_uniform_initializer(minval=0, maxval=0.5)

        out=conv3d_layer(
            "conv3d_API3D_",APM(inputs),self.conv2d.out_channels,
            kernel_size=self.kernel_dim,strides=self.stride,padding=self.padding,
            weight_initializer=init,use_bias=True
        )
        return out
        





class P3DA(object):
    def __init__(self, conv2d,time_stride=1,time_dim=3, **kwargs):
        super(P3DA, self).__init__()
        self.conv2d=conv2d
        self.time_dim=time_dim
        self.time_stride=time_stride
        self.kernel_dim=[1, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[1, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,0, conv2d.padding[0], conv2d.padding[1]]
        
    def build_network(self,inputs):
        # weight_2d=self.conv2d.weight.data
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d[:, :, 0, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)        
        spatial_conv3d=conv3d_layer(
            "P3DA_spatial_",inputs,self.conv2d.out_channels,
            kernel_size=self.kernel_dim,strides=self.stride,padding=self.padding,
            use_bias=True,weight_initializer=init
        )

        self.kernel_dim=[self.time_dim,1,1]
        self.stride=[self.time_stride,1,1]
        self.padding=[self.time_dim//2,0,0]

        # weight_2d=np.eye(self.conv2d.out_channels)
        # weight_2d=flow.expand_dims(weight_2d,axis=2)
        # weight_2d=flow.expand_dims(weight_2d,axis=2)
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d=np.tile(weight_3d,(1,1,self.time_dim,1,1))
        # middle_dix=self.time_dim//2
        # weight_3d[:, :, middle_idx, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)
        name=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        out=conv3d_layer(
            "P3DA_temporal_",spatial_conv3d,self.conv2d.out_channels,
            kernel_size=self.kernel_dim,strides=self.stride,padding=self.padding,
            weight_initializer=init,use_bias=False
        )
        return out


class P3DB(object):
    def __init__(self, conv2d,time_dim=3, time_stride=1, **kwargs):
        super(P3DB, self).__init__()
        self.conv2d=conv2d
        self.kernel_dim=[1, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[1, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,0, conv2d.padding[0], conv2d.padding[1]]
        self.time_dim=time_dim
        self.time_stride=time_stride
        
    def build_network(self,inputs):
        # weight_2d=self.conv2d.weight.data
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d[:, :, 0, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)
    
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)        
        out1=conv3d_layer(
            "P3DB_spatial_",inputs,self.conv2d.out_channels,
            kernel_size=self.kernel_dim,strides=self.stride,padding=self.padding,
            weight_initializer=init,use_bias=True
        )

        self.kernel_dim=[self.time_dim,1,1]
        self.stride=[self.time_stride, self.conv2d.stride[0], self.conv2d.stride[0]]
        self.padding=[self.time_dim//2,0,0]
        init=flow.constant_initializer(0)
        out2=conv3d_layer(
            "P3DB_temporal_",inputs, self.conv2d.out_channels,
            kernel_size=self.kernel_dim,strides=self.stride,padding=self.padding,
            use_bias=False,weight_initializer=init
        )
        out1= out1+out2
        return out1

class P3DC(object):
    def __init__(self, conv2d,  time_dim=3,time_stride=1,  **kwargs):
        super(P3DC, self).__init__()

        self.kernel_dim=[1, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[1, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,0, conv2d.padding[0], conv2d.padding[1]]
        self.conv2d=conv2d
        self.time_dim=time_dim
        self.time_stride=time_stride
    def build_network(self,inputs):
        # weight_2d=self.conv2d.weight.data
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d[:, :, 0, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)        
        out=conv3d_layer(
            "P3DC_spatial_",inputs,self.conv2d.out_channels,
            kernel_size=self.kernel_dim,strides=self.stride,padding=self.padding,
            use_bias=True,weight_initializer=init
        )

        self.kernel_dim=[self.time_dim, 1, 1]
        self.stride=[self.time_stride, 1, 1]
        self.padding=[self.time_dim//2, 0, 0]
        init=flow.constant_initializer(0)

        residual=conv3d_layer(
            "P3DC_temporal_",out,self.conv2d.out_channels,
            kernel_size=self.kernel_dim,strides=self.stride,padding=self.padding,
            use_bias=False,weight_initializer=init
        )
        out= out+residual
        return out


class APP3DA(object):
    def __init__(self, conv2d, time_dim=3, temperature=4, contrastive_att=True,time_stride=1):
        super(APP3DA, self).__init__()
        self.APM = APM(conv2d.out_channels, conv2d.out_channels//16, 
                       time_dim=time_dim, temperature=temperature, contrastive_att=contrastive_att)
        self.kernel_dim=[1, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[1, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,0, conv2d.padding[0], conv2d.padding[1]]
        self.conv2d=conv2d
        self.time_dim=time_dim
        self.time_stride=time_stride
    def build_network(self,inputs):
        # weight_2d = self.conv2d.weight.data
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d[:, :, 0, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)
        out=conv3d_layer(
            "APP3DA_spatial_",inputs, self.conv2d.out_channels, 
            kernel_size=self.kernel_dim, 
            strides=self.stride, padding=self.padding,use_bias=True,weight_initializer=init
        
        )
        self.kernel_dim=[self.time_dim, 1, 1]
        self.stride= [self.time_stride*self.time_dim, 1, 1]
        # weight_2d=np.eye(self.conv2d.out_channels)
        # weight_2d=flow.expand_dims(weight_2d,axis=2)
        # weight_2d=flow.expand_dims(weight_2d,axis=2)
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d=np.tile(weight_3d,(1,1,self.time_dim,1,1))
        # middle_idx = self.time_dim // 2
        # weight_3d[:, :, middle_idx, :, :] = weight_2d
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)
        #init=flow.constant_initializer(weight_3d)
        out=conv3d_layer(
    
            "APP3DA_temporal_",self.APM(out),self.conv2d.out_channels, 
            kernel_size=self.kernel_dim,
            strides=self.stride, padding="SAME",use_bias=False,weight_initializer=init
        )
        return out
class APP3DB(object):
    def __init__(self, conv2d, time_dim=3, temperature=4, contrastive_att=True,time_stride=1):
        super(APP3DB, self).__init__()
        self.APM = APM(conv2d.in_channels, conv2d.in_channels//16, 
                       time_dim=time_dim, temperature=temperature, contrastive_att=contrastive_att)
        self.kernel_dim=[1, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[1, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,0, conv2d.padding[0], conv2d.padding[1]]
        self.conv2d=conv2d
        self.time_dim=time_dim
        self.time_stride=time_stride
    def build_network(self,inputs):
        # weight_2d=self.conv2d.weight.data
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d[:, :, 0, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)

        out=conv3d_layer(
            "APP3DB_spatial_",inputs,self.conv2d.out_channels, 
            kernel_size=self.kernel_dim,
            strides=self.stride, padding=self.padding,use_bias=True,weight_initializer=init
        )
        
        self.kernel_dim=[self.time_dim,1,1]
        self.stride=[self.time_stride*self.time_dim,conv2d.stride[0],conv2d.stride[0]]
        init=flow.constant_initializer(0)
        out2=conv3d_layer(
            "APP3DB_temporal_",self.APM(inputs), self.conv2d.out_channels, 
            kernel_size=self.kernel_dim, strides=self.stride, 
            padding="SAME",use_bias=False,weight_initializer=init
        )
        out=out+out2
        return out

class APP3DC(object):
    def __init__(self, conv2d, time_dim=3,  temperature=4, contrastive_att=True,time_stride=1):
        super(APP3DC, self).__init__() 
        self.APM=APM(conv2d.out_channels, conv2d.out_channels//16, 
                       time_dim=time_dim, temperature=temperature, contrastive_att=contrastive_att)
        self.kernel_dim=[1, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[1, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,0, conv2d.padding[0], conv2d.padding[1]]
        self.time_dim=time_dim
        self.conv2d=conv2d
        self.time_stride=time_stride
    def build_network(self,inputs):
        # weight_2d=self.conv2d.weight.data
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d[:, :, 0, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)
        print("inputs",inputs.shape)
        out=conv3d_layer(
            "APP3DC_spatial_",inputs,self.conv2d.out_channels, 
            kernel_size=self.kernel_dim, strides=self.stride,
            padding=self.padding, use_bias=True,weight_initializer=init
        )

        self.kernel_dim=[self.time_dim,1,1]
        self.stride=[self.time_stride*self.time_dim,1,1]
        init=flow.constant_initializer(0)
        print("beforeAPM",out.shape)
        residual=conv3d_layer(
            "APP3DC_temporal_",self.APM.build_network(out),self.conv2d.out_channels, 
            kernel_size=self.kernel_dim, 
            strides=self.stride, padding=[0,0,0,0,0],use_bias=False,weight_initializer=init
        )
        print("afterAPM",residual.shape)

        out=out+residual
        return out