import os
import torch
import clip
import json
import random
import numpy as np

from pathlib import Path


def setup_seed(seed):
    if seed == 0:
        print('random seed')
        torch.backends.cudnn.benchmark = True
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)


def get_clip_feat_dim(clip_model, img=torch.ones((1, 3, 224, 224))):
    clip_model.eval()
    with torch.no_grad():
        output = clip_model.encode_image(img.cuda())
        print(f"{output.shape=}")
    return output.shape[1]


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as fp:
        return json.load(fp)


def log_init(logger, cfg):
    logger.info('**************************************************************')
    logger.info(f'Here are the args:')
    for arg in cfg.keys():
        logger.info(f'{arg} : {cfg[arg]}')


def make_dirs(*kargs):
    for dir in kargs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def make_dirs_from_file(*kargs):
    dirs = []
    for path in kargs:
        dirs.append(os.path.split(path)[0])
    make_dirs(*dirs)


def get_model_param_size(model):
    size = sum(param.numel() for param in model.parameters())
    return size


def save_model(save_dir, name, model, optimizer=None, epoch=0, lr_scheduler=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None
    }
    torch.save(checkpoint, os.path.join(save_dir, name))


def load_model(checkpoint_path, model, 
               optimizer=None, lr_scheduler=None, key_filter=lambda key: True):
    ckp = torch.load(checkpoint_path, map_location='cpu')
    net_state_dict = ckp['net']
    model_state_dict = model.state_dict()
    net_state_dict = {k: v for k, v in net_state_dict.items() if k in model_state_dict and key_filter(k)}
    model_state_dict.update(net_state_dict)
    model.load_state_dict(model_state_dict)
    if optimizer and ckp['optimizer']:
        optimizer.load_state_dict(ckp['optimizer'])
    if lr_scheduler and ckp['lr_scheduler']:
        lr_scheduler.load_state_dict(ckp['lr_scheduler'])
    return model.cuda(), optimizer, lr_scheduler, ckp['epoch']


def cal_acc_mid(logits, labels):
    pred = torch.argmax(logits, -1)
    acc_num = (pred == labels.cuda()).sum().item()
    total = len(labels)
    return acc_num, total

def cal_acc(logits, labels):
    acc_num, total = cal_acc_mid(logits, labels)
    acc = 1.0 * acc_num / total
    return acc

class AvgACC:
    def __init__(self) -> None:
        self.acc_num = 0
        self.total = 0
    
    def step(self, logits, labels):
        acc_num, total = cal_acc_mid(logits, labels)
        self.acc_num += acc_num
        self.total += total
    
    def cal(self):
        return 0.00 if self.total == 0 else 1.0 * self.acc_num / self.total


class my_scheduler:
    
    def __init__(self, optimizer, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0) -> None:
        self.optimizer = optimizer
        self.optimizer.param_groups[0]['lr'] = 0
        
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(0, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        self.schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        self.schedule = np.concatenate((warmup_schedule, self.schedule))
        self.id = 0
        assert len(self.schedule) == epochs * niter_per_ep
    
    def step(self):
        self.optimizer.param_groups[0]['lr'] = self.schedule[self.id]
        self.id += 1


# the following function is modified from Tip-Adapter

def clip_classifier(feat_path, classnames, template, clip_model):
    if os.path.exists(feat_path):
        print(f"Loading texture features from {feat_path}")
        text_feats = torch.load(feat_path, map_location='cpu')
        return text_feats.cuda()
    
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            if isinstance(template, list):
                texts = [t.format(classname) for t in template]
            elif isinstance(template, dict):
                texts = template[classname]
                
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
        
        make_dirs_from_file(feat_path)
        torch.save(clip_weights, feat_path)
            
    return clip_weights
