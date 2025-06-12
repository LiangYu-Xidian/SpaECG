import os
import torch
from datetime import datetime
import time
from torch.utils.tensorboard import SummaryWriter



class LossHistory():
    def __init__(self, log_dir, model):
        self.log_dir    = log_dir
        self.losses     = []
        # self.val_loss   = []
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        timestamp = datetime.fromtimestamp(time.time()).strftime('%m%d%H%M')
        self.prefix = "Train_"+timestamp
        self.writer = SummaryWriter(os.path.join(log_dir, self.prefix))
    #     try:
    # # --------- 1. tensorboard.SummaryWriter.add_graph记录model -------------#
    #         dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1],use_strict_trace=False)
    #         self.writer.add_graph(model, dummy_input)
    #     except:
    #         pass


    def append_loss(self, epoch, loss, loss_gene_adj, loss_cell_adj, loss_pred, loss_rec ):

    # --------- 2. 保存loss -------------#
        self.losses.append(loss)
        # self.val_loss.append(val_loss)

    # --------- 3. txt记录loss -------------#
        with open(os.path.join(self.log_dir, self.prefix+"/epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, self.prefix+"/epoch_loss_gene_adj.txt"), 'a') as f:
            f.write(str(loss_gene_adj))
            f.write("\n")
        with open(os.path.join(self.log_dir, self.prefix+"/epoch_loss_cell_adj.txt"), 'a') as f:
            f.write(str(loss_cell_adj))
            f.write("\n")
        with open(os.path.join(self.log_dir, self.prefix+"/epoch_loss_pred.txt"), 'a') as f:
            f.write(str(loss_pred))
            f.write("\n")
        with open(os.path.join(self.log_dir, self.prefix+"/epoch_loss_rec.txt"), 'a') as f:
            f.write(str(loss_rec))
            f.write("\n")
    # --------- 4. tensorboard.SummaryWriter.add_scalar记录loss -------------#
        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('loss_gene_adj', loss_gene_adj, epoch)
        self.writer.add_scalar('loss_cell_adj', loss_cell_adj, epoch)
        self.writer.add_scalar('loss_pred', loss_pred, epoch)
        self.writer.add_scalar('loss_rec', loss_rec, epoch)

