import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys
from collections import OrderedDict
from modelling.vtn_att_poseflow_model import VTNHCPF_Three_View, MMTensorNorm
from modelling.mvit_v2 import MVitV2_ThreeView

class MaxPool3dSamePadding(nn.MaxPool3d):
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)
    

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                output_channels,
                kernel_shape=(1, 1, 1),
                stride=(1, 1, 1),
                padding=0,
                activation_fn=F.relu,
                use_batch_norm=True,
                use_bias=False,
                name='unit_3d'):
    
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

            
    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x



class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                            name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                            name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                            name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                            name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                            name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                            name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):    
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,**kwargs):
        """Initializes I3D model instance.
        Args:
            num_classes: The number of outputs in the logit layer (default 400, which
                matches the Kinetics dataset).
            spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
                before returning (default True).
            final_endpoint: The model contains many possible endpoints.
                `final_endpoint` specifies the last endpoint for the model to be built
                up to. In addition to the output at `final_endpoint`, all the outputs
                at endpoints up to `final_endpoint` will also be returned, in a
                dictionary. `final_endpoint` must be one of
                InceptionI3d.VALID_ENDPOINTS (default 'Logits').
            name: A string (optional). The name of this module.
        Raises:
            ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        print("Model: InceptionI3d")
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                            padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                    name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                    name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                            padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                            padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                            padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                    stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                            kernel_shape=[1, 1, 1],
                            padding=0,
                            activation_fn=None,
                            use_batch_norm=False,
                            use_bias=True,
                            name='logits')

        self.build()
       
    def count(self):
           
        total = sum(p.numel() for p in self.parameters())
        print(f"Total params: {total}")

        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable params: {num_trainable_params}")
        
        print("Trainable params / Total params",num_trainable_params/total)

    def replace_logits(self, num_classes):
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=num_classes,
                            kernel_shape=[1, 1, 1],
                            padding=0,
                            activation_fn=None,
                            use_batch_norm=False,
                            use_bias=True,
                            name='logits')
        print("Reset to ",num_classes)
        
    
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, clip = None):
        x = clip.permute(0,2,1,3,4) # b,t,c,h,w -> b,c,t,h,w
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel
        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3).squeeze(-1)
        # logits is batch X time X classes, which is what we want to work with
        return {
            'logits':logits
        }
        

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
    


class InceptionI3D_ThreeView(nn.Module):
    def __init__(self, num_classes=199, spatial_squeeze=True,
                final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,**kwargs):
        super(InceptionI3D_ThreeView, self).__init__()
        print("Model: InceptionI3D_ThreeView")
        self.center = InceptionI3d(num_classes,spatial_squeeze,final_endpoint,name,in_channels,dropout_keep_prob)
        self.left = InceptionI3d(num_classes,spatial_squeeze,final_endpoint,name,in_channels,dropout_keep_prob)
        self.right = InceptionI3d(num_classes,spatial_squeeze,final_endpoint,name,in_channels,dropout_keep_prob)
        self.classififer = nn.Sequential(
            nn.Linear(3*(384+384+128+128),1024),
            nn.ReLU(),
            nn.Dropout(dropout_keep_prob),
            nn.Linear(1024,num_classes)
        )

    def remove_head(self):
        self.center.logits = nn.Identity()
        self.left.logits = nn.Identity()
        self.right.logits = nn.Identity()
    
    def freeze_and_remove(self,enpoints = 10):
        VALID_ENDPOINTS = self.center.VALID_ENDPOINTS
        print("Endpoints ",len(VALID_ENDPOINTS))
        for i in range(enpoints):
            for param in self.center.end_points[VALID_ENDPOINTS[i]].parameters():
                param.requires_grad = False
            for param in self.left.end_points[VALID_ENDPOINTS[i]].parameters():
                param.requires_grad = False
            for param in self.right.end_points[VALID_ENDPOINTS[i]].parameters():
                param.requires_grad = False
    
    def forward_features(self, left,center,right):    
        # b,t,x,c,h,w
        b,t,c,h,w = left.shape
        left_ft = self.left.extract_features(left.permute(0,2,1,3,4))
        center_ft = self.center.extract_features(center.permute(0,2,1,3,4))
        right_ft = self.right.extract_features(right.permute(0,2,1,3,4))
        
        output_features = torch.cat([left_ft,center_ft,right_ft],dim = -1)

        return output_features
        
    def forward(self, left,center,right):    
        # b,t,x,c,h,w
        b,t,c,h,w = left.shape
        left_ft = self.left.extract_features(left.permute(0,2,1,3,4))
        center_ft = self.center.extract_features(center.permute(0,2,1,3,4))
        right_ft = self.right.extract_features(right.permute(0,2,1,3,4))
        
        output_features = torch.cat([left_ft,center_ft,right_ft],dim = -1)
       

        y = self.classififer(output_features)

        return {
            'logits':y
        }
    
class InceptionI3D_HandCrop(nn.Module):
    def __init__(self, num_classes=199, spatial_squeeze=True,
                final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,**kwargs):
        super(InceptionI3D_HandCrop, self).__init__()
        print("Model: InceptionI3D_HandCrop")
        self.left = InceptionI3d(num_classes,spatial_squeeze,final_endpoint,name,in_channels,dropout_keep_prob)
        self.right = InceptionI3d(num_classes,spatial_squeeze,final_endpoint,name,in_channels,dropout_keep_prob)
        self.classififer = nn.Sequential(
            nn.Linear(2*(384+384+128+128),1024),
            nn.ReLU(),
            nn.Dropout(dropout_keep_prob),
            nn.Linear(1024,num_classes)
        )
        print("InceptionI3D_HandCrop")
        print("Num classes",num_classes)

    def remove_head(self):
        self.left.logits = nn.Identity()
        self.right.logits = nn.Identity()
    
    def freeze_and_remove(self,enpoints = 0):
        VALID_ENDPOINTS = self.left.VALID_ENDPOINTS
        print("Endpoints ",len(VALID_ENDPOINTS))
        for i in range(enpoints):
            for param in self.left.end_points[VALID_ENDPOINTS[i]].parameters():
                param.requires_grad = False
            for param in self.right.end_points[VALID_ENDPOINTS[i]].parameters():
                param.requires_grad = False
       
        
    def forward(self, clip = None,poseflow = None,**kwargs):    
       
        b,t,x,c,h,w = clip.shape
        left = clip[:,:,0] # b,t,c,h,w
        right = clip[:,:,1] # b,t,c,h,w
        left_ft = self.left.extract_features(left.permute(0,2,1,3,4))
        right_ft = self.right.extract_features(right.permute(0,2,1,3,4))
        
        output_features = torch.cat([left_ft,right_ft],dim = -1)
       

        y = self.classififer(output_features)

        return {
            'logits':y
        }
        
class I3D_OneView_Sim_Knowledge_Distillation(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                freeze_layers=0, dropout=0,spatial_squeeze=True,final_endpoint='Logits', name='inception_i3d', 
                in_channels=3, dropout_keep_prob=0.5, **kwargs):
        super().__init__()
        print("Model: I3D_OneView_Sim_Knowledge_Distillation V1")
        self.teacher = MVitV2_ThreeView(num_classes = num_classes)
        self.teacher.remove_head()  
        self.teacher.load_state_dict(torch.load("checkpoints/MVitV2_ThreeView/MVIT V2 Small for three view finetune from one view/best_checkpoints.pth",map_location='cpu'))
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.student = InceptionI3d(num_classes,spatial_squeeze,final_endpoint,name,in_channels,dropout_keep_prob)
        state_dict = torch.load('checkpoints/InceptionI3d/I3D for AUTSL pretrained on charades for one view/autsl_best_checkpoints.pth',map_location=torch.device('cpu'))
        self.student.replace_logits(226) # AUTSL
        self.student.load_state_dict(state_dict)
        self.student.logits = nn.Identity()
        self.norm = MMTensorNorm(-1)
        self.projection = nn.Linear(384+384+128+128,768*3)
           
        
    def forward(self,left = None,center = None,right = None):  
        b, t, c, h, w = left.size()
        self.teacher.eval()
        teacher_features = None
        y = None
        teacher_features = self.teacher.forward_features(left = left,center = center,right = right)
        
        student_features = self.student.extract_features(center.permute(0,2,1,3,4))
        student_features = self.projection(self.norm(student_features))

        if not self.training:
            y = self.teacher.classififer(student_features)
        
        return {
            'logits':y,
            'student_features': student_features,
            'teacher_features': teacher_features,
        }

class I3D_OneView_Sim_Knowledge_Distillation_V2(nn.Module):
    def __init__(self, num_classes=400, spatial_squeeze=True,
                final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,**kwargs):
        super().__init__()
        print("Model: I3D_OneView_Sim_Knowledge_Distillation V2")
        self.teacher = InceptionI3D_ThreeView(num_classes,spatial_squeeze,final_endpoint,name,in_channels,dropout_keep_prob)
        self.teacher.remove_head()  
        self.teacher.load_state_dict(torch.load("checkpoints/InceptionI3d_ThreeView/I3D for VN SIGN for three view finetune form oneview/best_checkpoints.pth",map_location='cpu'))
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.student = InceptionI3d(num_classes,spatial_squeeze,final_endpoint,name,in_channels,dropout_keep_prob)
        state_dict = torch.load('checkpoints/InceptionI3d/I3D for AUTSL pretrained on charades for one view/autsl_best_checkpoints.pth',map_location=torch.device('cpu'))
        self.student.replace_logits(226) # AUTSL
        self.student.load_state_dict(state_dict)
        self.student.replace_logits(199)
       
       
    def forward(self,left = None,center = None,right = None):  
        b, t, c, h, w = left.size()
        self.teacher.eval()
        teacher_logits = None
        student_logits = None
        y = None
       
        teacher_logits = self.teacher(left = left,center = center,right = right)['logits']
        student_logits = self.student(clip = center)['logits']
        

        if not self.training:
            y = student_logits
        
        
        return {
            'logits':y,
            'teacher_logits': teacher_logits,
            'student_logits': student_logits,
        }

class I3D_OneView_Sim_Knowledge_Distillation_Inference(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                freeze_layers=0, dropout=0,spatial_squeeze=True,final_endpoint='Logits', name='inception_i3d', 
                in_channels=3, dropout_keep_prob=0.5, **kwargs):
        super().__init__()
        print("Model: I3D_OneView_Sim_Knowledge_Distillation_Inference")
        print("*"*20)
        self.student = InceptionI3d(num_classes,spatial_squeeze,final_endpoint,name,in_channels,dropout_keep_prob)
        self.student.logits = nn.Identity()
        self.norm = MMTensorNorm(-1)
        self.projection = nn.Linear(384+384+128+128,768*3)
        self.classififer = nn.Sequential(
            nn.Linear(3*768,1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024,num_classes)
        )
        print("*"*20)
           
        
    def forward(self,clip = None): 
        center = clip
        b,t,c,h,w = center.shape
        assert c == 3
        
        student_features = self.student.extract_features(center.permute(0,2,1,3,4))
        student_features = self.projection(self.norm(student_features))

        y = self.classififer(student_features)
        
        return {
            'logits':y
        }

class InceptionI3D_ThreeView_ShareWeights(nn.Module):
    def __init__(self, num_classes=199, spatial_squeeze=True,
                final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,**kwargs):
        super(InceptionI3D_ThreeView_ShareWeights, self).__init__()
        print("Model: InceptionI3D_ThreeView_ShareWeights")
        print("*"*20)
        self.encoder = InceptionI3d(num_classes,spatial_squeeze,final_endpoint,name,in_channels,dropout_keep_prob)
        self.left_pj = nn.Linear(384+384+128+128,384+384+128+128)
        self.right_pj = nn.Linear(384+384+128+128,384+384+128+128)
        self.center_pj = nn.Linear(384+384+128+128,384+384+128+128)
        self.classififer = nn.Sequential(
            nn.Linear(3*(384+384+128+128),1024),
            nn.ReLU(),
            nn.Dropout(dropout_keep_prob),
            nn.Linear(1024,num_classes)
        )
        print("*"*20)

    def remove_head(self):
        self.encoder.logits = nn.Identity()
    
    def freeze_and_remove(self,enpoints = 10):
        VALID_ENDPOINTS = self.encoder.VALID_ENDPOINTS
        print("Endpoints ",len(VALID_ENDPOINTS))
        for i in range(enpoints):
            for param in self.encoder.end_points[VALID_ENDPOINTS[i]].parameters():
                param.requires_grad = False
    
    def count(self):
           
        total = sum(p.numel() for p in self.parameters())
        print(f"Total params: {total}")

        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable params: {num_trainable_params}")
        
        print("Trainable params / Total params",num_trainable_params/total)

       
        
    def forward(self, left,center,right):    
        # b,t,x,c,h,w
        b,t,c,h,w = left.shape
        left_ft = self.encoder.extract_features(left.permute(0,2,1,3,4))
        center_ft = self.encoder.extract_features(center.permute(0,2,1,3,4))
        right_ft = self.encoder.extract_features(right.permute(0,2,1,3,4))

        # fc learn corresponding views
        left_ft = self.left_pj(left_ft)
        right_ft = self.right_pj(right_ft)
        center_ft = self.center_pj(center_ft)
        
        output_features = torch.cat([left_ft,center_ft,right_ft],dim = -1)
       

        y = self.classififer(output_features)

        return {
            'logits':y
        }