import torch.nn as nn
import torch.optim as optim
from modelling.vtn_att_poseflow_model import VTNHCPF,VTNHCPF_Three_View,VTNHCPF_OneView_Sim_Knowledge_Distilation,VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference,VTNHCPF_Three_View_Visual_Prompt_Tuning
import torch
from trainer.tools import MyCustomLoss,OLM_Loss
from modelling.gcn_bert import GCN_BERT
from modelling.i3d import InceptionI3d,InceptionI3D_ThreeView,InceptionI3D_HandCrop,I3D_OneView_Sim_Knowledge_Distillation,I3D_OneView_Sim_Knowledge_Distillation_Inference,InceptionI3D_ThreeView_ShareWeights,I3D_OneView_Sim_Knowledge_Distillation_V2
from modelling.videomae import pretrain_videomae_small_patch16_224,vit_small_patch16_224
from torchvision import models
from torch.nn import functional as F
from modelling.swin_transformer import SwinTransformer3d,SwinTransformer3d_ThreeView,SwinTransformer3d_HandCrop,VideoSwinTransformer_OneView_Sim_Knowledge_Distillation,VideoSwinTransformer_OneView_Sim_Knowledge_Distillation_Inference,SwinTransformer3d_ThreeView_ShareWeights
from collections import OrderedDict
from modelling.mvit_v2 import mvit_v2_s,MVitV2_ThreeView,MVitV2_HandCrop,MvitV2_OneView_Sim_Knowledge_Distillation,MvitV2_OneView_Sim_Knowledge_Distillation_Inference,MVitV2_ThreeView_ShareWeights,MVitV2_ThreeView_ShareWeights_Visual_Prompt_Tuning,MvitV2_OneView_KD_Knowledge_Distillation_Visual_Prompt_Tuning


def load_criterion(train_cfg):
    criterion = None
    if train_cfg['criterion'] == "MyCustomLoss":
        criterion = MyCustomLoss(label_smoothing=train_cfg['label_smoothing'])
    if train_cfg['criterion'] == "OLM_Loss": 
        criterion = OLM_Loss(label_smoothing=train_cfg['label_smoothing'])
    assert criterion is not None
    return criterion

def load_optimizer(train_cfg,model):
    optimzer = None
    if train_cfg['optimzer'] == "SGD":
        optimzer = optim.SGD(model.parameters(), lr=train_cfg['learning_rate'],weight_decay=float(train_cfg['w_decay']),momentum=0.9,nesterov=True)
    if train_cfg['optimzer'] == "Adam":
        optimzer = optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'],weight_decay=float(train_cfg['w_decay']))
    assert optimzer is not None
    return optimzer

def load_lr_scheduler(train_cfg,optimizer):
    scheduler = None
    if train_cfg['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=train_cfg['scheduler_factor'], patience=train_cfg['scheduler_patience'])
    if train_cfg['lr_scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg['lr_step_size'], gamma=train_cfg['gamma'])
    assert scheduler is not None
    return scheduler

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    try:
        if m.weight is not None:
            m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    except:
        pass


def load_model(cfg):
    if cfg['training']['pretrained']:
        print(f"load pretrained model: {cfg['training']['pretrained_model']}")

        if cfg['data']['model_name'] == 'vtn_att_poseflow':
            if '.ckpt' in cfg['training']['pretrained_model']:
                model = VTNHCPF(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
                new_state_dict = {}
                for key, value in torch.load(cfg['training']['pretrained_model'],map_location='cpu')['state_dict'].items():
                        new_state_dict[key.replace('model.','')] = value
                model.reset_head(226) # AUTSL
                model.load_state_dict(new_state_dict)
                model.reset_head(model.num_classes)
            else:
                model = VTNHCPF(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
        elif cfg['data']['model_name'] == 'VTNHCPF_Three_view':
            model = VTNHCPF_Three_View(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            model.add_backbone()
            model.remove_head_and_backbone()
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
        
            print("Load VTNHCPF Three View")
        elif cfg['data']['model_name'] == 'VTNHCPF_Three_View_Visual_Prompt_Tuning':
            model = VTNHCPF_Three_View_Visual_Prompt_Tuning(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
           
        elif cfg['data']['model_name'] == 'InceptionI3d':
            model = InceptionI3d(**cfg['model'])
            if "autsl" in cfg['training']['pretrained_model'].split("/")[-1]:
                model.replace_logits(226)
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu')))
                model.replace_logits(model._num_classes)
                print("Finetune fron AUTSL checkpoint")
            elif "Knowledge_Distillation" in cfg['training']['pretrained_model']:
                state_dict = torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu'))
                new_state_dict = {}
                for key,value in state_dict.items():
                    if key.startswith('teacher'): # omit teacher state dict
                        continue
                    new_state_dict[key.replace('student.',"")] = value
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu')))

        elif cfg['data']['model_name'] == 'InceptionI3d_ThreeView':
            model = InceptionI3D_ThreeView(**cfg['model'])
            model.remove_head()
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu')),strict = True)
            print("Load pretrained I3D Three View")

        elif cfg['data']['model_name'] == 'InceptionI3D_ThreeView_ShareWeights':
            model = InceptionI3D_ThreeView_ShareWeights(**cfg['model'])
            model.remove_head()
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu')),strict = True)
            print("Load InceptionI3D_ThreeView_ShareWeights")
        elif cfg['data']['model_name'] == 'InceptionI3D_HandCrop':
            model = InceptionI3D_HandCrop(**cfg['model'])
            if "autsl" in cfg['training']['pretrained_model'].split("/")[-1]:
                model.right.replace_logits(226)
                model.left.replace_logits(226)
                model.right.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu')),strict = True)
                model.left.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu')),strict = True)
                model.remove_head()
                model.freeze_and_remove(enpoints=0)
                print("Load I3D Hand Crop Pretrained on AUTSL")
            else:
                model.remove_head()
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu')),strict = True)
                print("Load I3D Hand Crop")
        elif cfg['data']['model_name'] == 'videomae':
            if cfg['training']['stage'] == 1:
                model = pretrain_videomae_small_patch16_224()
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu'))['model'])
            else:
                model = vit_small_patch16_224(**cfg['model'])
                state_dict = torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu'))
                new_state_dict = {}
                for key,value in state_dict.items():
                    if 'encoder.norm.weight' in key or 'encoder.norm.bias' in key:
                        continue
                    if key.startswith('encoder.'):
                        new_state_dict[key.replace('encoder.','')] =value
                model.load_state_dict(new_state_dict,strict = False)
                
        elif cfg['data']['model_name'] == 'swin_transformer':
            model = SwinTransformer3d(**cfg['model'])
            state_dict = torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu'))
            if "autsl" in cfg['training']['pretrained_model'].split("/")[-1]:
                model.reset_head(226)
                model.load_state_dict(state_dict)
                model.reset_head(model.num_classes)
            else:
                model.load_state_dict(state_dict)

        elif cfg['data']['model_name'] == 'swin_transformer_3d_ThreeView':
            model = SwinTransformer3d_ThreeView(**cfg['model'])
            model.remove_head()
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
            print("Load Video Swin Transformer for Three view")

        elif cfg['data']['model_name'] == 'SwinTransformer3d_ThreeView_ShareWeights':
            model = SwinTransformer3d_ThreeView_ShareWeights(**cfg['model'])
            model.remove_head()
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
            print("Load SwinTransformer3d_ThreeView_ShareWeights model")
        elif cfg['data']['model_name'] == 'SwinTransformer3d_HandCrop':
            model = SwinTransformer3d_HandCrop(**cfg['model'])
            state_dict = torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu'))
            if "autsl" in cfg['training']['pretrained_model'].split("/")[-1]:
                model.right.reset_head(226)
                model.left.reset_head(226)
                model.right.load_state_dict(state_dict,strict = True)
                model.left.load_state_dict(state_dict,strict = True)
                model.remove_head()
                model.freeze_and_remove(layers=4)
            else:
                model.remove_head()
                model.load_state_dict(state_dict,strict = True)

        elif cfg['data']['model_name'] == 'mvit_v2':
            model = mvit_v2_s(**cfg['model'])
            state_dict = torch.load(cfg['training']['pretrained_model'],map_location=torch.device('cpu'))
            if "autsl" in cfg['training']['pretrained_model'].split("/")[-1]:
                model.reset_head(226)
                model.load_state_dict(state_dict)
                model.reset_head(model.num_classes)
            else:
                model.load_state_dict(state_dict)

        elif cfg['data']['model_name'] == 'MVitV2_ThreeView':
            model = MVitV2_ThreeView(**cfg['model'])

            model.remove_head()
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
            print("Load Mvit V2 Three View")

        elif cfg['data']['model_name'] == 'MVitV2_ThreeView_ShareWeights':
            model = MVitV2_ThreeView_ShareWeights(**cfg['model'])
            model.remove_head()
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
            print("Load Mvit V2 Three View Share Weights")
        elif cfg['data']['model_name'] == 'MVitV2_ThreeView_ShareWeights_Visual_Prompt_Tuning':
            model = MVitV2_ThreeView_ShareWeights_Visual_Prompt_Tuning(**cfg['model'])
            model.remove_head()
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
            print("Load MVitV2_ThreeView_ShareWeights_Visual_Prompt_Tuning")
        elif cfg['data']['model_name'] == 'MVitV2_HandCrop':
            model = MVitV2_HandCrop(**cfg['model'])
            if "autsl" in cfg['training']['pretrained_model'].split("/")[-1]:
                pass
            else:
                model.remove_head()
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
    else:
        if cfg['data']['model_name'] == 'vtn_att_poseflow':
            model = VTNHCPF(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
        if cfg['data']['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation':
            model = VTNHCPF_OneView_Sim_Knowledge_Distilation(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
        if cfg['data']['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference':
            model = VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            state_dict = torch.load("checkpoints/VTNHCPF_OneView_Sim_Knowledge_Distilation/VTNHCPF_OneView_Sim_Knowledge_Distilation/best_checkpoints.pth",map_location='cpu')
            new_state_dict = {}
            for key,value in state_dict.items():
                if key.startswith('teacher'): # omit teacher state dict
                    if key.split('.')[1].startswith('classifier'): # save the classifier of the teacher model
                        new_state_dict[key.replace('teacher.','')] = value
                    continue
                new_state_dict[key] = value
            model.load_state_dict(new_state_dict)

        elif cfg['data']['model_name'] == 'VTNHCPF_Three_view':
            model = VTNHCPF_Three_View(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            state_dict = torch.load("/mnt/disk4/handsign_project/son_data/Experiment/checkpoints/vtn_att_poseflow/vtn_att_poseflow autsl to vn sign for one view (1-1000)/best_checkpoints.pth",map_location='cpu')
            model.center.load_state_dict(state_dict,strict = True)
            model.right.load_state_dict(state_dict,strict = True)
            model.left.load_state_dict(state_dict,strict = True)
            model.add_backbone()
            model.remove_head_and_backbone()
            model.freeze(layers = 0)
            print("Load VTNHCPF Three View")
        elif cfg['data']['model_name'] == 'VTNHCPF_Three_View_Visual_Prompt_Tuning':
            model = VTNHCPF_Three_View_Visual_Prompt_Tuning(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            state_dict = torch.load("checkpoints/vtn_att_poseflow/vtn_att_poseflow autsl to vn sign for one view (1- 1000)/best_checkpoints.pth",map_location='cpu')
            model.encoder.load_state_dict(state_dict,strict = False) # ingore prompt and head
            model.count()

        elif cfg['data']['model_name'] == 'InceptionI3d':
            model = InceptionI3d(**cfg['model'])
            new_dict = {}
            for key,value in torch.load('pretrained_models/InceptionI3D/rgb_charades.pt',map_location=torch.device('cpu')).items():
                if key.startswith('logits'):
                    continue
                new_dict[key] = value
            model.load_state_dict(new_dict,strict = False)

        elif cfg['data']['model_name'] == 'InceptionI3d_ThreeView':
            model = InceptionI3D_ThreeView(**cfg['model']) #checkpoints/InceptionI3d/I3D finetune pretrained from AUTSL for one view with Blur video/best_checkpoints.pth
            state_dict = torch.load("checkpoints/InceptionI3d/I3D finetune pretrained from AUTSL for one view with Blur video/best_checkpoints.pth",map_location='cpu')
            model.center.load_state_dict(state_dict,strict = True)
            model.right.load_state_dict(state_dict,strict = True)
            model.left.load_state_dict(state_dict,strict = True)
            model.remove_head()
            model.freeze_and_remove(enpoints=6)
            print("Load I3D Three View")

        elif cfg['data']['model_name'] == 'InceptionI3D_ThreeView_ShareWeights':
            model = InceptionI3D_ThreeView_ShareWeights(**cfg['model'])
            state_dict = torch.load("checkpoints/InceptionI3d/I3D finetune from autsl for one view/best_checkpoints.pth",map_location='cpu')
            model.encoder.load_state_dict(state_dict,strict = True)
            model.remove_head()
            model.freeze_and_remove(enpoints=6)
            print("Load InceptionI3D_ThreeView_ShareWeights")
        elif cfg['data']['model_name'] == 'InceptionI3D_HandCrop':
            model = InceptionI3D_HandCrop(**cfg['model'])
            new_dict = {}
            for key,value in torch.load('pretrained_models/InceptionI3D/rgb_charades.pt',map_location=torch.device('cpu')).items():
                if key.startswith('logits'):
                    continue
                new_dict[key] = value
            model.remove_head()
            model.right.load_state_dict(new_dict,strict = True)
            model.left.load_state_dict(new_dict,strict = True)
            model.freeze_and_remove(enpoints=0)
            print("Load I3D Hand Crop")
        elif cfg['data']['model_name'] == 'I3D_OneView_Sim_Knowledge_Distillation':
            # model = I3D_OneView_Sim_Knowledge_Distillation(**cfg['model'])
            model = I3D_OneView_Sim_Knowledge_Distillation_V2(**cfg['model'])
        elif cfg['data']['model_name'] == 'I3D_OneView_Sim_Knowledge_Distillation_Inference':
            model = I3D_OneView_Sim_Knowledge_Distillation_Inference(**cfg['model'])
            state_dict = torch.load("checkpoints/I3D_OneView_Sim_Knowledge_Distillation/I3D_OneView_Sim_Knowledge_Distillation_v1/best_checkpoints.pth",map_location='cpu')
            new_state_dict = {}
            for key,value in state_dict.items():
                if key.startswith('teacher'): # omit teacher state dict
                    if key.split('.')[1].startswith('classififer'): # save the classifier of the teacher model
                        new_state_dict[key.replace('teacher.','')] = value
                    continue
                new_state_dict[key] = value
            model.load_state_dict(new_state_dict)
        
        elif cfg['data']['model_name'] == 'videomae':
            model = pretrain_videomae_small_patch16_224()
            
        elif cfg['data']['model_name'] == 'swin_transformer':
            model = SwinTransformer3d(**cfg['model'])
            weights = models.video.Swin3D_T_Weights.DEFAULT.get_state_dict(progress=True)
            model.reset_head(400)
            model.load_state_dict(weights)
            model.reset_head(model.num_classes)

        elif cfg['data']['model_name'] == 'swin_transformer_3d_ThreeView':
            model = SwinTransformer3d_ThreeView(**cfg['model'])
            state_dict = torch.load("checkpoints/swin_transformer/Swin Transformer 3D Tiny for one view pretrained from AUTSL (1-1000)/best_checkpoints.pth",map_location='cpu')
            model.center.load_state_dict(state_dict,strict = True)
            model.right.load_state_dict(state_dict,strict = True)
            model.left.load_state_dict(state_dict,strict = True)
            model.remove_head()
            model.freeze_and_remove(layers=4)
        elif cfg['data']['model_name'] == 'SwinTransformer3d_ThreeView_ShareWeights':
            model = SwinTransformer3d_ThreeView_ShareWeights(**cfg['model'])
            state_dict = torch.load("checkpoints/swin_transformer/Swin Transformer 3D Tiny for one view finetune from autsl /best_checkpoints.pth",map_location='cpu')
            model.encoder.load_state_dict(state_dict,strict = True)
            model.remove_head()
            model.freeze_and_remove(layers=4)
            print("Load SwinTransformer3d_ThreeView_ShareWeights model")
        
        elif cfg['data']['model_name'] == 'SwinTransformer3d_HandCrop':
            model = SwinTransformer3d_HandCrop(**cfg['model'])
            state_dict  = models.video.Swin3D_T_Weights.DEFAULT.get_state_dict(progress=True)
            model.right.reset_head(400)
            model.left.reset_head(400)
            model.right.load_state_dict(state_dict,strict = True)
            model.left.load_state_dict(state_dict,strict = True)
            model.remove_head()
            model.freeze_and_remove(layers=4)
        elif cfg['data']['model_name'] == 'VideoSwinTransformer_OneView_Sim_Knowledge_Distillation':
            model = VideoSwinTransformer_OneView_Sim_Knowledge_Distillation(**cfg['model'])
        elif cfg['data']['model_name'] == 'VideoSwinTransformer_OneView_Sim_Knowledge_Distillation_Inference':
            model = VideoSwinTransformer_OneView_Sim_Knowledge_Distillation_Inference(**cfg['model'])
            state_dict = torch.load("checkpoints/VideoSwinTransformer_OneView_Sim_Knowledge_Distillation/VideoSwinTransformer_OneView_Sim_Knowledge_Distillation/best_checkpoints.pth",map_location='cpu')
            new_state_dict = {}
            for key,value in state_dict.items():
                if key.startswith('teacher'): # omit teacher state dict
                    if key.split('.')[1].startswith('classififer'): # save the classifier of the teacher model
                        new_state_dict[key.replace('teacher.','')] = value
                    continue
                new_state_dict[key] = value
            model.load_state_dict(new_state_dict)
        elif cfg['data']['model_name'] == 'mvit_v2':
            model = mvit_v2_s(**cfg['model'])
            model.reset_head(400)
            weights = models.video.MViT_V2_S_Weights.KINETICS400_V1.get_state_dict(progress=True)
            model.load_state_dict(weights)
            model.reset_head(model.num_classes)

        elif cfg['data']['model_name'] == 'MVitV2_ThreeView':
            model = MVitV2_ThreeView(**cfg['model'])
            state_dict = torch.load("checkpoints/mvit_v2/MVIT V2 Small for one view (1-1000) pretrained from AUTSL for Blur video/best_checkpoints.pth",map_location='cpu')
            model.center.load_state_dict(state_dict,strict = True)
            model.right.load_state_dict(state_dict,strict = True)
            model.left.load_state_dict(state_dict,strict = True)
            model.remove_head()
            model.freeze_and_remove(layers=8)
        elif cfg['data']['model_name'] == 'MVitV2_ThreeView_ShareWeights':
            model = MVitV2_ThreeView_ShareWeights(**cfg['model'])
            state_dict = torch.load("checkpoints/mvit_v2/MVIT V2 Small for one view finetune from AUTSL/best_checkpoints.pth",map_location='cpu')
            model.encoder.load_state_dict(state_dict,strict = True)
            model.remove_head()
            model.freeze_layers(8)
        elif cfg['data']['model_name'] == 'MVitV2_ThreeView_ShareWeights_Visual_Prompt_Tuning':
            model = MVitV2_ThreeView_ShareWeights_Visual_Prompt_Tuning(**cfg['model'])
            state_dict = torch.load("checkpoints/mvit_v2/MVIT V2 Small for one view (1-1000) finetune from AUTSL/best_checkpoints.pth",map_location='cpu')
            model.encoder.load_state_dict(state_dict,strict = False)
            model.remove_head()
            model.count()
        elif cfg['data']['model_name'] == 'MVitV2_HandCrop':
            model = MVitV2_HandCrop(**cfg['model'])
            state_dict = models.video.MViT_V2_S_Weights.KINETICS400_V1.get_state_dict(progress=True)
            model.left.reset_head(400)
            model.right.reset_head(400)
            model.right.load_state_dict(state_dict,strict = True)
            model.left.load_state_dict(state_dict,strict = True)
            model.remove_head()
            model.freeze_and_remove(layers=8)
        elif cfg['data']['model_name'] == 'MvitV2_OneView_Sim_Knowledge_Distillation':
            model = MvitV2_OneView_Sim_Knowledge_Distillation(**cfg['model'])
        elif cfg['data']['model_name'] == 'MvitV2_OneView_KD_Knowledge_Distillation_Visual_Prompt_Tuning':
             model = MvitV2_OneView_KD_Knowledge_Distillation_Visual_Prompt_Tuning(**cfg['model'])
        elif cfg['data']['model_name'] == 'MvitV2_OneView_Sim_Knowledge_Distillation_Inference':
            model = MvitV2_OneView_Sim_Knowledge_Distillation_Inference(**cfg['model'])
            state_dict = torch.load("checkpoints/MvitV2_OneView_Sim_Knowledge_Distillation/MvitV2_OneView_Sim_Knowledge_Distillation/best_checkpoints.pth",map_location='cpu')
            new_state_dict = {}
            for key,value in state_dict.items():
                if key.startswith('teacher'): # omit teacher state dict
                    if key.split('.')[1].startswith('classififer'): # save the classifier of the teacher model
                        new_state_dict[key.replace('teacher.','')] = value
                    continue
                new_state_dict[key] = value
            model.load_state_dict(new_state_dict)
        
        
  




    assert model is not None
    print("loaded model")
    return model
        