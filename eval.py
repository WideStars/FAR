import torch
from tqdm import tqdm

from utils import AvgACC, cal_acc


def fuse_logits(mlp_logits, clip_logits, beta=1.0):
    return beta * mlp_logits + (1 - beta) * clip_logits


class Eval:
    
    def __init__(self, cfg, clip_model, val_loader, text_feats, logger) -> None:
        self.cfg = cfg
        self.clip_model = clip_model
        self.text_feats = text_feats
        self.val_loader = val_loader
        self.logger = logger
        self.batch_size = cfg['batch_size']
        
    def evaluate_epoch(self, images):    
        image_feats, mlp_logits, _ = self.clip_model.encode_image(images)
        image_feats /= image_feats.norm(dim=-1, keepdim=True)
        clip_logits = 100. * image_feats @ self.text_feats
        return clip_logits, mlp_logits

    def eval(self, use_beta=None):
        ACC = AvgACC()
        self.clip_model.eval()
        all_clip_logits = []
        all_mlp_logits = []
        all_labels = []
        with torch.no_grad():
            with tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc='Evaluate') as tqdm_eval:
                for _, (images, labels) in tqdm_eval:
                    clip_logits, mlp_logits = self.evaluate_epoch(images.cuda())
                    ACC.step(mlp_logits, labels)
                    all_clip_logits.append(clip_logits)
                    all_mlp_logits.append(mlp_logits)
                    all_labels.append(labels)
        
        self.all_clip_logits = torch.cat(all_clip_logits, dim=0)
        self.all_mlp_logits = torch.cat(all_mlp_logits, dim=0)
        self.all_labels = torch.cat(all_labels, dim=0)
        
        best_beta, last_acc, best_acc = self.search_hp()
        if use_beta:
            logits = fuse_logits(self.all_mlp_logits, self.all_clip_logits, use_beta)
            acc = cal_acc(logits, self.all_labels) * 100.
            self.logger.info(f"{self.cfg['desc']} :*** valdation best beta = {use_beta:.4f} => {acc:.2f}% [{last_acc=}, {best_acc=}]")
        # return ACC.cal() * 100.0
        return best_beta
    
    def search_hp(self):
        start = self.cfg['search_low']
        end = self.cfg['search_high']
        step = self.cfg['search_step']
        beta_list = [i * (end - start) / step + start for i in range(step + 1)]
        best_beta, best_acc = start, 0.
        accs = []
        for beta in beta_list:
            self.beta = beta
            logits = fuse_logits(self.all_mlp_logits, self.all_clip_logits, beta)
            acc = cal_acc(logits, self.all_labels) * 100.
            accs.append((beta, acc))
            if acc > best_acc:
                best_acc = acc
                best_beta = beta
        
        if 'imagenet' in self.cfg['dataset']:
            self.logger.info(f"{self.cfg['desc']}:*** last acc => {accs[-1][-1]:.2f}%, best acc => {best_acc:.2f}% (beta = {best_beta:.4f}), accs => {accs}")
        return best_beta, accs[-1][-1], best_acc