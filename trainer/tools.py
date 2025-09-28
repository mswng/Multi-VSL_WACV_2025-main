import numpy as np
import torch
from torch import nn
import math
from einops import rearrange
import torch.nn.functional as F
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta: # avoid score == self.best_score + self.delta and it is constant throughout the epochs
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


        
class MyCustomLoss(nn.Module):

    def __init__(self, reduction=None,label_smoothing = 0,cls_x = 1,cts_x = 1,cosine_x = 1):
        """
        Args:
            reduction: defined for compatibility with other losses.
        """
        super(MyCustomLoss, self).__init__()
        print("Use Label Smoothing: ",label_smoothing)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.mse  = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
       
       

    def forward(self,logits = None,labels = None,student_features = None,teacher_features = None,student_logits = None,teacher_logits = None,**kwargs):
        loss = 0
        loss_dict = {
           
        }
       
        if student_features is not None:
            mse_loss = self.mse(student_features,teacher_features)
            loss = loss + mse_loss 
            loss_dict['mse_loss'] = mse_loss.item()

        if student_logits is not None:
            student_logits = F.log_softmax(student_logits,dim = 1)
            teacher_logits = teacher_logits.softmax(dim = 1)
            kl_loss = self.kl_loss(student_logits, teacher_logits)
            loss = loss + kl_loss 
            loss_dict['kl_loss'] = kl_loss.item()


        if logits is not None:

            classification_loss = self.criterion(logits,labels)
            loss = loss + classification_loss    
            loss_dict['classification_loss'] = classification_loss.item()
        
     
        # return loss/n_loss
        return loss,loss_dict
        


class OLM_Loss(nn.Module):

    def __init__(self, reduction=None,label_smoothing = 0):
        """
        Args:
            reduction: defined for compatibility with other losses.
        """
        super(OLM_Loss, self).__init__()
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.weights = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
        self.optimizer = torch.optim.Adam([self.weights],lr = 1e-3)
        self.initial_loss = None

    def forward(self, logits_v,logists_k,logist_mt,labels,iteration,
                    **kwargs):
        loss = 0
        loss_dict = {
           
        }
        logits = torch.stack([logits_v, logists_k, logist_mt])
        loss_v =  self.criterion( logits_v, labels)
        loss_k =  self.criterion( logists_k, labels)
        loss_mt =  self.criterion(  logist_mt, labels)
        loss_dict['vision_cls'] = loss_v.item()
        loss_dict['keypoints_cls'] = loss_k.item()
        loss_dict['classification_loss'] = loss_mt.item()
        if iteration  == 0:
            self.initial_loss = torch.stack([loss_v, loss_k, loss_mt]).detach()
        loss = self.weights[0] *loss_v + self.weights[1] *loss_k + self.weights[2] *loss_mt
       

        logits_norm = [self.weights[i] * torch.norm(logit, dim=-1).detach() for i, logit in enumerate(logits)]
        # Optimize logit coefficients
        logits_norm = torch.stack(logits_norm, dim=-1)
        loss_ratio = torch.stack([loss_v, loss_k, loss_mt]).detach() / self.initial_loss
        rt = loss_ratio / loss_ratio.mean()
        logits_norm_avg = logits_norm.mean(-1).detach()
        constant = (logits_norm_avg.unsqueeze(-1) @ rt.unsqueeze(0)).detach()
        logitsnorm_loss = torch.abs(logits_norm - constant).sum()

       
        loss_dict['logitsnorm_loss'] = logitsnorm_loss.item()
        loss_dict['vision_w'] = self.weights[0].item()
        loss_dict['keypoints_w'] = self.weights[1].item()
        loss_dict['fusion_w'] = self.weights[2].item()
        loss_dict['iteration'] = iteration

        return loss,loss_dict,logitsnorm_loss