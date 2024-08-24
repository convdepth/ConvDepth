
import torch
import torch.nn as nn
import torch.nn.functional as F

def refine_motion_field(motion_filed,layer,align_corners,scope=None):
    _,h,w,_=torch.unbind(torch.size(layer))
    upsampled_motion_field=F.interpolate(motion_filed,[h,w],mode='bilinear',align_corners=False)
    conv_input=torch.cat([upsampled_motion_field,layer],3)
    conv_output=nn.Conv2d(conv_input, max(4, layer.shape.as_list()[-1]), [3, 3], stride=1)
    conv_input = nn.Conv2d(conv_input, max(4, layer.shape.as_list()[-1]), [3, 3], stride=1)
    conv_output2=nn.Conv2d(conv_input, max(4, layer.shape.as_list()[-1]), [3, 3], stride=1)
    conv_output3=torch.cat([conv_output, conv_output2], axis=-1)
    conv_output=nn.Conv2d(conv_output3,motion_filed.shape.as_list()[-1], [1, 1],stride=1,)
    return upsampled_motion_field+conv_output

def motion_field_net(images,weight_reg=0.0,
                     align_corners=True,
                     auto_mask=False):
    conv1 = nn.Conv2d(images, 16, [3, 3], stride=2,)
    conv2 = nn.Conv2d(conv1, 32, [3, 3], stride=2, )
    conv3 = nn.Conv2d(conv2, 64, [3, 3], stride=2, )
    conv4 = nn.Conv2d(conv3, 128, [3, 3], stride=2, )
    conv5 = nn.Conv2d(conv4, 256, [3, 3], stride=2, )
    conv6 = nn.Conv2d(conv5, 512, [3, 3], stride=2, scope='Conv6')
    conv7 = nn.Conv2d(conv6, 1024, [3, 3], stride=2, scope='Conv7')
    bottleneck = torch.mean(conv7, axis=[1, 2], keepdims=True)
    background_motion = nn.Conv2d(
        bottleneck,
        6, [1, 1],
        stride=1,
        activation_fn=None,
        biases_initializer=None,
        scope='background_motion')
