from utils.video_augmentation import *
from dataset.vtn_att_poseflow_model_dataset import VTN_ATT_PF_Dataset
from .three_viewpoints import ThreeViewsData
from .gcn_bert import GCN_BERT
from dataset.i3d import InceptionI3D_Data
from dataset.videomae import VideoMAE
from dataset.swin_transformer import SwinTransformer
from dataset.mvit import MVIT
from dataset.vtn_hc_pf_three_view import VTNHCPF_ThreeViewsData
from dataset.distilation import Distilation

def build_video_transform(dataset_cfg,split):
    if split == 'train':
        transform = Compose(
                            Scale(dataset_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7),
                            MultiScaleCrop((dataset_cfg['vid_transform']['IMAGE_SIZE'], dataset_cfg['vid_transform']['IMAGE_SIZE']), scales),
                            # CenterCrop(dataset_cfg['vid_transform']['IMAGE_SIZE']),
                            # RandomHorizontalFlip(), 
                            Resize(dataset_cfg['vid_transform']['IMAGE_SIZE']),
                            RandomVerticalFlip(),
                            RandomRotate(p=0.3),
                            RandomShear(0.3,0.3,p = 0.3),
                            Salt( p = 0.5),
                            GaussianBlur( sigma=1,p = 0.5),
                            ColorJitter(0.5, 0.5, 0.5,p = 0.5),
                            ToFloatTensor(), PermuteImage(),
                            Normalize(dataset_cfg['vid_transform']['NORM_MEAN_IMGNET'],dataset_cfg['vid_transform']['NORM_STD_IMGNET']))
    else:
        transform = Compose(
                            Scale(dataset_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7), 
                            CenterCrop(dataset_cfg['vid_transform']['IMAGE_SIZE']), 
                            # Resize(dataset_cfg['vid_transform']['IMAGE_SIZE']), => three views
                            ToFloatTensor(),
                            PermuteImage(),
                            Normalize(dataset_cfg['vid_transform']['NORM_MEAN_IMGNET'],dataset_cfg['vid_transform']['NORM_STD_IMGNET']))
    return transform

def build_image_transform(dataset_cfg,split,model = None):
    if split == 'train':
        if model is not None:
            data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
            transform = timm.data.create_transform(**data_cfg)
    else:
        if model is not None:
            data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
            transform = timm.data.create_transform(**data_cfg)
    assert transform is not None
    return transform

def build_dataset(dataset_cfg, split,model = None,**kwargs):
    dataset = None

    if dataset_cfg['model_name'] == "vtn_att_poseflow" or 'HandCrop' in dataset_cfg['model_name'] or dataset_cfg['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference':
        dataset = VTN_ATT_PF_Dataset(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    if dataset_cfg['model_name'] == "gcn_bert": 
        dataset = GCN_BERT(dataset_cfg['base_url'],split,None,dataset_cfg,**kwargs)

    distillation_models = ['MvitV2_OneView_Sim_Knowledge_Distillation','I3D_OneView_Sim_Knowledge_Distillation','VideoSwinTransformer_OneView_Sim_Knowledge_Distillation','MvitV2_OneView_KD_Knowledge_Distillation_Visual_Prompt_Tuning']
    
    if 'ThreeView' in dataset_cfg['model_name'] or dataset_cfg['model_name'] in distillation_models:
        dataset = ThreeViewsData(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    if dataset_cfg['model_name'] == "InceptionI3d" or dataset_cfg['model_name'] == 'I3D_OneView_Sim_Knowledge_Distillation_Inference':
        dataset = InceptionI3D_Data(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    if dataset_cfg['model_name'] == "videomae" :
        dataset = VideoMAE(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)
    
    if dataset_cfg['model_name'] == "swin_transformer" or dataset_cfg['model_name'] == 'VideoSwinTransformer_OneView_Sim_Knowledge_Distillation_Inference':
        dataset = SwinTransformer(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)
        
    if 'mvit' in dataset_cfg['model_name'] or dataset_cfg['model_name'] == 'MvitV2_OneView_Sim_Knowledge_Distillation_Inference':
        dataset = MVIT(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    if dataset_cfg['model_name'] == 'VTNHCPF_Three_view' or dataset_cfg['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation' or dataset_cfg['model_name'] ==  "VTNHCPF_Three_View_Visual_Prompt_Tuning":
        dataset = VTNHCPF_ThreeViewsData(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)
    
    

    # if dataset_cfg['model_name'] in distilation_models:
    #     dataset = Distilation(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)
    

    assert dataset is not None
    return dataset



    