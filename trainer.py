import sys
import time
import torch
from tqdm import tqdm
import torch.nn.functional as F

from utils import my_scheduler, AvgACC, save_model
from eval import Eval

    
def clip_forward(clip_model, images, text_feats):
    image_feats, mlp_logits, new_feats = clip_model.encode_image(images.cuda())
    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
    clip_logits = 100. * image_feats @ text_feats
    return image_feats, clip_logits, mlp_logits, new_feats


class Trainer:
    
    def __init__(self, cfg, clip_model, train_loader, test_loader, logger, text_feats, val_loader=None):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.clip_model = clip_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.logger = logger
        self.checkpoint_dir = cfg['checkpoint_dir']
        self.save_interval = cfg['save_interval']
        self.epochs = cfg['train_epoch']
        self.log_dir = f"./log/{self.checkpoint_dir.split('/')[-1]}"
        
        self.optimizer = torch.optim.AdamW(self.clip_model.parameters(), lr=cfg['lr'], eps=1e-4)
        self.scheduler = my_scheduler(self.optimizer, cfg['lr'], 1e-6, self.epochs, len(self.train_loader), 10)
        
        self.text_feats = text_feats
        self.eval = Eval(self.cfg, self.clip_model, test_loader, self.text_feats, self.logger)
                    
    def get_loss(self, labels, clip_logits, mlp_logits, feats, new_feats):
        l1_loss = F.l1_loss(mlp_logits, clip_logits)
        kl_loss = -F.kl_div(new_feats.softmax(dim=-1).log(), feats.softmax(dim=-1), reduction='batchmean')
        ce_loss = F.cross_entropy(mlp_logits, labels)
        delta = self.cfg['delta']
        gammar = self.cfg['gammar']
        loss = delta * kl_loss + gammar * l1_loss + ce_loss
        return loss, [l1_loss, kl_loss, ce_loss]
    
    def train_mode(self):
        self.clip_model.visual.adapter.train()
        self.clip_model.visual.adapter_mlp.train()
        
    def train_epoch(self, epoch):
        self.train_mode()
        train_loss = 0.0
        ACC = AvgACC()
        loss_list = [0, 0, 0]
        
        with tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Training epoch {epoch}") as tqdm_train:
            for _, (images, labels) in tqdm_train:
                images, labels = images.cuda(), labels.cuda()
                feats, clip_logits, mlp_logits, new_feats = clip_forward(self.clip_model, images, self.text_feats)
                
                loss, losses = self.get_loss(labels, clip_logits, mlp_logits, feats, new_feats)

                if torch.isnan(loss):
                    self.logger.info(f"{self.cfg['desc']}:!!! Loss is NaN. Program terminated.")
                    sys.exit()

                ACC.step(mlp_logits, labels)
                train_loss += loss.item()
                for i, l in enumerate(losses):
                    loss_list[i] += l.item()
                tqdm_train.set_postfix(cur_loss=loss.item())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
            train_acc = ACC.cal()
            train_loss = train_loss / len(self.train_loader)
            
        print(f"{epoch=}, {loss_list=}")
        if epoch == self.epochs - 1:
            self.logger.info(f"[l1_loss, kl_loss, ce_loss] => {loss_list}")
            
        return train_acc * 100, train_loss
        
    def train(self):
        self.logger.info('-------------------- START TRAINING --------------------')
        train_name = self.logger.name
        train_st = time.time()
        self.validate()
        
        for epoch in range(self.epochs):
            epoch_st = time.time()
            self.logger.info(f'====> Epoch: {epoch}')
            train_acc, train_loss = self.train_epoch(epoch)
            epoch_ed = time.time()
            self.logger.info(f"      train_acc: {train_acc:.4f} %    train_loss: {train_loss:.4f}    train_time: {(epoch_ed - epoch_st):.4f} s    lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:.8f}")
            
            # if (epoch % cfg['save_interval'] == 0 and epoch != 0):
                # save_model(self.checkpoint_dir, f'{train_name}_epoch_{epoch}.pth', self.clip_model)
                # self.validate()
            if epoch == self.epochs - 1:
                # save_model(self.checkpoint_dir, f'{train_name}_last.pth', self.clip_model)
                self.validate()
        
        duration = int(time.time() - train_st)
        self.logger.info(f'Total time used for training: {duration // 3600} h {duration % 3600 // 60} min {duration % 60} sec')
        
    def validate(self):
        self.eval.clip_model = self.clip_model
        
        val_best_beta = None
        if self.val_loader:
            self.eval.val_loader = self.val_loader
            val_best_beta = self.eval.eval()
            
        self.eval.val_loader = self.test_loader
        self.eval.eval(use_beta=val_best_beta)
