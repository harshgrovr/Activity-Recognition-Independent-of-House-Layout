import numpy as np
import torch

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
    def __call__(self, val_f1, model, test_index, epoch):
        score = val_f1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_f1, model, test_index, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func('EarlyStopping counter: {} out of {}'.format(self.counter,  self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_f1, model, test_index, epoch)
            self.counter = 0

    def save_checkpoint(self, val_f1, model, test_index, epoch):
        '''Saves model when val F1 increase.'''
        print('\n epoch', epoch)
        if self.verbose:
            self.trace_func('Validation F1 Increased ({} --> {}).  Saving model ...'.format(self.val_loss_min, val_f1))
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_f1
