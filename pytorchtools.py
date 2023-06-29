import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False,dataset_name='Ga',model_cofig='',delta=0,args=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset_name=dataset_name
        self.model_config=model_cofig
        self.args=args
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        os.chdir('/home/coche/RushuangZhou_Phdstudent/Cinc2021data/result')
        if self.args.ema:
            ema_flag='ema'
        else:
            ema_flag='noema'
        if self.args.teacher_pretrain:
            pre_flag=''
        else:
            pre_flag='_pre'
        if self.model_config=='_ECGmatch':
            torch.save(model.state_dict(), self.dataset_name +self.model_config+'_'+str(int(10*self.args.Semi_loss_weight))+'_'+str(int(10*self.args.relationship_weight))+'_'+ema_flag+'_checkpoint_'+str(self.args.seed)+pre_flag+'.pkl') 
        else:
            torch.save(model.state_dict(), self.dataset_name +self.model_config+'_'+ema_flag+ '_checkpoint.pkl')  # 这里会存储迄今最优模型的参数
        # torch.save(model, 'finish_model.pkl') # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss