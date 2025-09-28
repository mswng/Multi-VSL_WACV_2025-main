import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm.auto import tqdm
from .tools import EarlyStopping
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import pickle
class Trainer:
    def __init__(self,model,criterion,optimizer,device,scheduler = None,top_k = 5,
                epoch = 100,logging = None,cfg = None,num_accumulation_steps = 1,
                patience = 7,verbose = True,delta = 0,is_early_stopping = True,gradient_clip_val = 1.0,
                log_train_step = True,log_steps = 100,evaluate_strategy = "epoch",evaluate_step = 50,wandb = None,k_fold = None
                ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.top_k = top_k
        self.epoch = epoch
        self.train_acc, self.val_acc = 0, 0
        self.train_losses,self.val_losses, self.train_accs, self.val_accs = [],[], [], []
        self.lr_progress = []
        self.top_train_acc, self.top_val_acc = 0, 0
        self.logging = logging
        self.cfg = cfg
        self.num_accumulation_steps = num_accumulation_steps
        if k_fold is None:
            self.early_stopping = EarlyStopping(patience=patience,verbose=verbose,delta = delta,
                                            path=f"checkpoints/{cfg['data']['model_name']}/" +cfg['training']['experiment_name'] + "/best_checkpoints" + ".pth")
        else:
            self.early_stopping = EarlyStopping(patience=patience,verbose=verbose,delta = delta,
                                            path=f"checkpoints/{cfg['data']['model_name']}/" +cfg['training']['experiment_name'] + f"/best_checkpoints_fold_{k_fold}" + ".pth")
        self.is_early_stopping = is_early_stopping
        self.gradient_clip_val = gradient_clip_val
        self.log_train_step = log_train_step
        self.log_steps = log_steps
        self.evaluate_strategy = evaluate_strategy
        self.evaluate_step  = evaluate_step
        self.wandb = wandb
        self.k_fold = k_fold
        self.test_accuracy = None
    
    def train(self,train_loader,val_loader,test_loader):
        cfg = self.cfg
        for epoch in tqdm(range(self.epoch)):
            if self.evaluate_strategy  == 'epoch':
                train_loss_log,train_loss, _, _, train_acc = self.train_epoch(train_loader,epoch=epoch)
                self.train_losses.append(train_loss / len(train_loader))
                self.train_accs.append(train_acc)
            else:
                train_loss_log,train_loss, _, _, train_acc,val_loss,_, _, val_acc = self.train_epoch(train_loader,val_loader,epoch)
            if self.k_fold is None:
                torch.save(self.model.state_dict(), 
                    f"checkpoints/{cfg['data']['model_name']}/" +cfg['training']['experiment_name'] + "/current_checkpoints" + ".pth")
            else:
                torch.save(self.model.state_dict(), 
                    f"checkpoints/{cfg['data']['model_name']}/" +cfg['training']['experiment_name'] + f"/current_checkpoints_fold_{self.k_fold}" + ".pth")
            if val_loader:
                if self.evaluate_strategy  == 'epoch':
                    val_loss_log,val_loss,_, _, val_acc = self.evaluate(val_loader,print_stats=self.cfg['training']['print_stats'],epoch=epoch)
                    self.val_losses.append(val_loss / len(val_loader))
                    self.val_accs.append(val_acc)
                if self.evaluate_strategy  == 'epoch':
                    self.early_stopping(val_loss = val_loss_log['classification_loss'],model = self.model)
            if self.scheduler:
                if self.cfg['training']['lr_scheduler'] == "StepLR":
                    self.scheduler.step()
                elif self.cfg['training']['lr_scheduler'] == 'ReduceLROnPlateau':
                    self.scheduler.step(val_loss / len(val_loader))     

            

            if epoch % cfg['training']['log_freq'] == 0:
                print("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss / len(train_loader)) + " acc: " + str(train_acc))
                self.logging.info("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss / len(train_loader)) + " acc: " + str(train_acc))
                self.logging.info("[" + str(epoch + 1) + "] TRAIN  loss dict: " + str(train_loss_log))

                if val_loader:
                    print("[" + str(epoch + 1) + "]" + f" VALIDATION loss: {val_loss/len(val_loader)}" + " VALIDATION  acc: " + str(val_acc))
                    self.logging.info("[" + str(epoch + 1) + "]" + f" VALIDATION loss: {val_loss/len(val_loader)}" + " VALIDATION  acc: " + str(val_acc))
                    self.logging.info("[" + str(epoch + 1) + "] VALIDATION  loss dict: " + str(val_loss_log))

                print("")
                self.logging.info("")


            self.lr_progress.append(self.optimizer.param_groups[0]["lr"])
            if self.is_early_stopping and self.early_stopping.early_stop:
                print("\n\n***Stop training***\n\n")
                self.logging.info("\n\n***Stop training***\n\n")
                break
            
            if self.k_fold is None:
                self.wandb.log({
                    "Loss": self.wandb.plot.line_series(
                        xs=range(len(self.train_losses)),
                        ys=[self.train_losses,self.val_losses],
                        keys= ["Train Loss","Val Loss"],
                        title="Loss",
                        xname="x epochs"
                    ),
                    "Accuracy": self.wandb.plot.line_series(
                        xs=range(len(self.train_accs)),
                        ys=[self.train_accs,self.val_accs],
                        keys=["Train Accuracy", "Valiation Accuracy"],
                        title="Accuracy",
                        xname="x epochs"),
                })
            else:
                self.wandb.log({
                    f"Loss Fold {self.k_fold}": self.wandb.plot.line_series(
                        xs=range(len(self.train_losses)),
                        ys=[self.train_losses,self.val_losses],
                        keys= [f"Train Loss Fold {self.k_fold}",f"Val Loss Fold {self.k_fold}"],
                        title=f"Loss Fold {self.k_fold}",
                        xname="x epochs"
                    ),
                    f"Accuracy Fold {self.k_fold}": self.wandb.plot.line_series(
                        xs=range(len(self.train_accs)),
                        ys=[self.train_accs,self.val_accs],
                        keys=[f"Train Accuracy Fold {self.k_fold}", f"Valiation Accuracy Fold {self.k_fold}"],
                        title=f"Accuracy Fold {self.k_fold}",
                        xname="x epochs"),
                })

        

        # MARK: TESTING

        print("\nTesting checkpointed models starting...\n")
        self.logging.info("\nTesting checkpointed models starting...\n")
        if test_loader:
            if self.k_fold is None:
                self.model.load_state_dict(torch.load(f"checkpoints/{cfg['data']['model_name']}/" +cfg['training']['experiment_name'] + "/best_checkpoints" + ".pth"))
            else:
                self.model.load_state_dict(torch.load(f"checkpoints/{cfg['data']['model_name']}/" +cfg['training']['experiment_name'] + f"/best_checkpoints_fold_{self.k_fold}" + ".pth"))
            _,_, _,_,  eval_acc = self.evaluate(test_loader, print_stats=True,epoch = 0)

            print("\nTesting accuracy:" , eval_acc)
            self.test_accuracy = eval_acc
            self.logging.info("\nTesting accuracy: " + str(eval_acc))
        if self.k_fold is None:
            self.wandb.run.finish()
            
    def train_epoch(self,dataloader,val_loader = None,epoch = None):
        self.model.train()
        pred_correct, pred_all = 0, 0
        running_loss = 0.0
        loss_log = None
        n_step_loss_log = None
        self.optimizer.zero_grad()
        n_step_loss = 0
        val_loss = 0
        val_acc = 0
        val_pred_correct = 0
        val_pred_all = 0
        self.optimizer.zero_grad()  
        if self.cfg['training']['criterion'] == "OLM_Loss":
            self.criterion.optimizer.zero_grad()  
        
        

        for idx, data in enumerate(tqdm(dataloader)):
            
            inputs, labels = data
            inputs = {key:values.to(self.device,non_blocking=True) for key,values in inputs.items() }
            labels = labels.to(self.device,non_blocking=True)
            

            outputs = self.model(**inputs)
            if self.cfg['training']['criterion'] == "OLM_Loss":
                loss,loss_dict,logitsnorm_loss = self.criterion(**outputs, labels=labels,iteration = epoch)
                logitsnorm_loss = logitsnorm_loss / self.num_accumulation_steps
                logitsnorm_loss.backward()
            else:
                loss,loss_dict = self.criterion(**outputs, labels=labels,epoch = epoch)

            if loss_log is None:
                loss_log = loss_dict
            else:
                loss_log = {key:value + loss_dict[key] for key,value in loss_log.items()}
            
            if n_step_loss_log is None:
                n_step_loss_log = loss_dict
            else:
                n_step_loss_log = {key:value + loss_dict[key] for key,value in n_step_loss_log.items()}
            
            running_loss += loss.item()
            n_step_loss += loss.item()
            
            loss = loss / self.num_accumulation_steps
            loss.backward()
            
            if self.log_train_step and (idx+1) % self.log_steps == 0 :
                n_step_loss_log = {key:value/self.log_steps for key,value in n_step_loss_log.items()}
                self.logging.info(f"Step[{idx+1}/{len(dataloader)}]: training loss : {n_step_loss/self.log_steps} TRAIN  loss dict:  {str(n_step_loss_log)}")
                
                n_step_loss = 0
                n_step_loss_log = None
                
            if ((idx + 1) % self.num_accumulation_steps == 0) or (idx + 1 == len(dataloader)):
                # clip grad
                if self.gradient_clip_val != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_val)
                # Update Optimizer
                self.optimizer.step()
                if self.cfg['training']['criterion'] == "OLM_Loss":
                    self.criterion.optimizer.step()  
                    self.criterion.optimizer.zero_grad()  
                self.optimizer.zero_grad()  
                
            if (idx+1) % self.evaluate_step == 0 and self.evaluate_strategy == 'step':
                if self.is_early_stopping and self.early_stopping.early_stop:
                    continue
                val_loss,val_pred_correct, val_pred_all, val_acc = self.evaluate(val_loader,print_stats=self.cfg['training']['print_stats'])
                self.early_stopping(val_loss=val_loss / len(val_loader),model = self.model)
                self.logging.info(f"Step[{idx+1}/{len(dataloader)}]: Evalutation: Val loss: {val_loss/len(val_loader)} ----- Val accuracy: {val_acc}")
                self.model.train()
                self.val_losses.append(val_loss / len(val_loader))
                self.val_accs.append(val_acc)
                self.train_losses.append(running_loss / (idx+1))
            # Statistics
            if outputs['logits'] is not None:
                logits = outputs['logits']
                pred_correct += (logits.argmax(dim = -1) == labels).sum().item()
                    
                pred_all += labels.shape[0]
        if outputs['logits'] is None:
                pred_correct = 0
                pred_all = 1   

        loss_log = {key:value/len(dataloader) for key,value in loss_log.items()}
        
        if self.evaluate_strategy == 'step':
            return loss_log,running_loss, pred_correct, pred_all, (pred_correct / pred_all),val_loss,val_pred_correct, val_pred_all, val_acc
        
        return loss_log,running_loss, pred_correct, pred_all, (pred_correct / pred_all)

    def evaluate(self, dataloader, print_stats=True, epoch=None):
        self.model.eval()
        loss_log = None
        pred_correct, pred_all = 0, 0
        running_loss = 0.0
        pred = []
        gr_th = []
        stats = {i: [0, 0] for i in range(self.cfg['model']['num_classes'])}
        results = pd.DataFrame(columns=['label', 'prediction'])

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader)):
                inputs, labels = data
                inputs = {key: values.to(self.device, non_blocking=True) for key, values in inputs.items()}
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(**inputs)

                if self.cfg['training']['criterion'] == "OLM_Loss":
                    loss, loss_dict, logitsnorm_loss = self.criterion(**outputs, labels=labels, iteration=i)
                else:
                    loss, loss_dict = self.criterion(**outputs, labels=labels, epoch=epoch)

                if loss_log is None:
                    loss_log = loss_dict
                else:
                    loss_log = {key: value + loss_dict[key] for key, value in loss_log.items()}
                running_loss += loss.item()

                if outputs['logits'] is not None:
                    logits = outputs['logits']
                    prediction = logits.argmax(dim=-1)
                    pred_all += labels.shape[0]
                    pred_correct += (prediction == labels).sum().item()

                    # Add to DataFrame
                    batch_results = pd.DataFrame({
                        'label': labels.cpu().numpy(),
                        'prediction': prediction.cpu().numpy()
                    })
                    results = pd.concat([results, batch_results], ignore_index=True)

                    for idx in range(labels.shape[0]):
                        if labels[idx].item() == prediction[idx]:
                            stats[labels[idx].item()][0] += 1
                        stats[labels[idx].item()][1] += 1
                    
                    pred.extend(prediction.cpu().numpy().tolist())
                    gr_th.extend(labels.cpu().numpy().tolist())
                    
            if print_stats and outputs['logits'] is not None:
                stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
                print("Label accuracies statistics:")
                print(str(stats) + "\n")
                self.logging.info("Label accuracies statistics:")
                self.logging.info(str(stats) + "\n")

            if outputs['logits'] is None:
                pred_correct = 0
                pred_all = 1
        # Save results to CSV after loop
        results.to_csv('label_predictions_1view.csv', index=False)

        # Existing logic to return loss and accuracy
        loss_log = {key: value / len(dataloader) for key, value in loss_log.items()}
        return loss_log, running_loss, pred_correct, pred_all, (pred_correct / pred_all)


    # def evaluate(self,dataloader, print_stats=True,epoch = None):
    #     self.model.eval()
    #     loss_log = None
    #     pred_correct, pred_all = 0, 0
    #     running_loss = 0.0
    #     pred = []
    #     gr_th = []
    #     stats = {i: [0, 0] for i in range(self.cfg['model']['num_classes'])}
    #     with torch.no_grad():
    #         for i, data in enumerate(tqdm(dataloader)):
    #             inputs, labels = data
    #             inputs = {key:values.to(self.device,non_blocking=True)  for key,values in inputs.items() }
    #             labels = labels.to(self.device,non_blocking=True)
    #             outputs = self.model(**inputs)
    #             # compute loss

    #             if self.cfg['training']['criterion'] == "OLM_Loss":
    #                 loss,loss_dict,logitsnorm_loss = self.criterion(**outputs, labels=labels,iteration = i)
    #             else:
    #                 loss,loss_dict = self.criterion(**outputs, labels=labels,epoch = epoch)

    #             if loss_log is None:
    #                 loss_log = loss_dict
    #             else:
    #                 loss_log = {key:value + loss_dict[key] for key,value in loss_log.items()}
    #             running_loss += loss.item()
                
    #             if outputs['logits'] is not None:
    #                 logits = outputs['logits']
    #                 prediction = logits.argmax(dim = -1)
    #                 pred_all += labels.shape[0]
    #                 pred_correct += (logits.argmax(dim = -1) == labels).sum().item()
                                
    #                 for idx in range(labels.shape[0]):
    #                     if labels[idx].item() == prediction[idx]:
    #                         stats[labels[idx].item()][0] += 1
    #                     stats[labels[idx].item()][1] += 1
                    
    #                 pred.extend(prediction.cpu().numpy().tolist())
    #                 gr_th.extend(labels.cpu().numpy().tolist())
    #             print(inputs)
            
    #         if print_stats and outputs['logits'] is not None:
    #             stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
    #             print("Label accuracies statistics:")
    #             print(str(stats) + "\n")
    #             self.logging.info("Label accuracies statistics:")
    #             self.logging.info(str(stats) + "\n")

    #         if outputs['logits'] is None:
    #             pred_correct = 0
    #             pred_all = 1
    #     loss_log = {key:value/len(dataloader) for key,value in loss_log.items()}
    #     return loss_log,running_loss,pred_correct, pred_all, (pred_correct / pred_all)

    def evaluate_top_k(self, dataloader):
        pred_correct, pred_all = 0, 0

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader)):
                inputs, labels = data
                # Đảm bảo dữ liệu được chuyển đến thiết bị tính toán đúng cách
                inputs = {key: values.to(self.device, non_blocking=True) for key, values in inputs.items()}
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True).reshape(-1,)

                # Lấy đầu ra từ mô hình
                outputs = self.model(**inputs)
                # Đảm bảo rằng 'logits' là một tensor
                logits = outputs['logits']  # Chỉnh sửa ở đây để 'logits' là tensor, không phải list

                # Sử dụng topk để lấy các chỉ số của các dự đoán hàng đầu
                top_k_predictions = torch.topk(logits, self.top_k).indices.tolist()

                # Tính số lượng dự đoán đúng
                for idx in range(labels.shape[0]):
                    if labels[idx].item() in top_k_predictions[idx]:
                        pred_correct += 1
                
                # Tính tổng số lượng dự đoán
                pred_all += labels.shape[0]

        # Tính và trả về độ chính xác
        return pred_correct, pred_all, (pred_correct / pred_all)
    
    def evaluate_per_class(self, dataloader):
        class_correct = {i: 0 for i in range(self.cfg['model']['num_classes'])}
        class_total = {i: 0 for i in range(self.cfg['model']['num_classes'])}

        with torch.no_grad():
            for _, data in enumerate(tqdm(dataloader)):
                inputs, labels = data
                inputs = {key: values.to(self.device, non_blocking=True) for key, values in inputs.items()}
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(**inputs)
                logits = outputs['logits']
                predictions = logits.argmax(dim=-1)

                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        class_correct[label.item()] += 1
                    class_total[label.item()] += 1

        per_class_accuracy = [class_correct[cls] / class_total[cls] if class_total[cls] != 0 else 0 for cls in class_total]
        average_accuracy = sum(per_class_accuracy) / len(per_class_accuracy)
        return average_accuracy


    def evaluate_top_k_per_class(self, dataloader):
        class_correct = {i: 0 for i in range(self.cfg['model']['num_classes'])}
        class_total = {i: 0 for i in range(self.cfg['model']['num_classes'])}

        with torch.no_grad():
            for _, data in enumerate(tqdm(dataloader)):
                inputs, labels = data
                inputs = {key: values.to(self.device, non_blocking=True) for key, values in inputs.items()}
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True).reshape(-1,)

                outputs = self.model(**inputs)
                logits = outputs['logits']
                top_k_preds = torch.topk(logits, self.top_k).indices

                for idx, label in enumerate(labels):
                    if label.item() in top_k_preds[idx]:
                        class_correct[label.item()] += 1
                    class_total[label.item()] += 1

        per_class_top_k_accuracy = [class_correct[cls] / class_total[cls] if class_total[cls] != 0 else 0 for cls in class_total]
        average_top_k_accuracy = sum(per_class_top_k_accuracy) / len(per_class_top_k_accuracy)
        return average_top_k_accuracy

    def plot_loss(self,split='train'):
        fig, ax = plt.subplots()
        if split == 'train':
            # train
            ax.plot(range(1, len(self.train_losses) + 1), self.train_losses, c="#D64436", label="Training loss")
        else:
            # val
            ax.plot(range(1, len(self.val_losses) + 1), self.val_losses, c="#4DAF4A", label="Validation loss")
        
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set(xlabel="Epoch", ylabel="Loss", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
        ax.grid()

        fig.savefig("out-img/" + f"{self.cfg['data']['model_name']}/{self.cfg['training']['experiment_name']}/" + f"{split}_loss.png")
        
        print("\nStatistics have been plotted.\nThe experiment is finished.")
        self.logging.info("\nStatistics have been plotted.\nThe experiment is finished.")
    
    def plot_acc(self,split='train'):
        fig, ax = plt.subplots()
        if split == 'train':
            # train
            ax.plot(range(1, len(self.train_accs) + 1), self.train_accs, c="#00B09B", label="Training accuracy")
        else:
            # val
            ax.plot(range(1, len(self.val_accs) + 1), self.val_accs, c="#E0A938", label="Validation accuracy")
        
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set(xlabel="", ylabel="Accuracy", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
        ax.grid()

        fig.savefig("out-img/" + f"{self.cfg['data']['model_name']}/{self.cfg['training']['experiment_name']}/" + f"{split}_accuracy.png")
        
        print("\nStatistics have been plotted.\nThe experiment is finished.")
        self.logging.info("\nStatistics have been plotted.\nThe experiment is finished.")
    
    def plot_lr(self):
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, len(self.lr_progress) + 1), self.lr_progress, label="LR")
        ax1.set(xlabel="Epoch", ylabel="lr", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
        ax1.grid()
        fig1.savefig("out-img/" + f"{self.cfg['data']['model_name']}/{self.cfg['training']['experiment_name']}/" + "lr.png")
        print("\nStatistics have been plotted.\nThe experiment is finished.")
        self.logging.info("\nStatistics have been plotted.\nThe experiment is finished.")

    