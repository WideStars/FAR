import os
import argparse
import yaml

import torch

import clip
from logger import *
from trainer import Trainer
from datasets import build_dataset
from datasets.utils import _transform, build_data_loader
from utils import setup_seed, make_dirs, clip_classifier


def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings in yaml format')
    
    parser.add_argument('--shots', default=16, type=int, help='number of shots for each class in training')
    parser.add_argument('--train_epoch', default=100, type=int, help='number of epochs to train the model')
    parser.add_argument('--title', type=str, default='default_title', help='title of this training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning_rate')
    parser.add_argument('--log_file', default='log', type=str, help='log file')
    parser.add_argument('--desc', default='default description', type=str, help='more details and description of this training')
    
    parser.add_argument('--backbone', type=str, choices=['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'], default='RN50', help='backbone of the visual endoer')
    parser.add_argument('--seed', default=1, type=int, help='seed for the whole training')
    parser.add_argument('--delta', default=1, type=float, help='weight for the -kl loss')
    parser.add_argument('--gammar', default=1, type=float, help='weight for the l1 loss')
    
    args = parser.parse_args()
    
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    for arg in vars(args):
        cfg[arg] = getattr(args, arg)
    torch.set_num_threads(cfg['num_threads'])

    return cfg


def main():

    # Load config file
    cfg = get_arguments()
    setup_seed(cfg['seed'])
    
    # CLIP: download clip model to ./model/clip
    clip_model, preprocess = clip.load(cfg['backbone'], download_root='./model/clip', num_classes=cfg['num_classes'])
    clip_model.eval()
    
    # Get dataset and dataloader
    print(f"Preparing {cfg['dataset']} dataset.")
    train_transform = _transform(224, True)
    dataset = build_dataset(cfg['dataset'], cfg['data_path'], cfg['shots'])
    train_loader = build_data_loader(data_source=dataset.train_x, batch_size=cfg['batch_size'], tfm=train_transform, is_train=True, shuffle=True)
    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    
    # Show config
    assert cfg['num_classes'] == len(dataset.classnames)
    print("\nRunning configs.")
    print(cfg, "\n")
    
    # Initialize diretory and logger
    dir_name = f"s{cfg['shots']}_{cfg['dataset']}_{cfg['backbone']}"
    checkpoint_dir = f'./checkpoint/{dir_name}'
    cfg['checkpoint_dir'] = checkpoint_dir
    log_dir = f'./log/{dir_name}'
    make_dirs(checkpoint_dir, log_dir)
    setup_logging(save_dir=log_dir, file_name=cfg['log_file'])
    logger = logging.getLogger(name=cfg['title'])
    # log_init(logger, cfg)    

    # Load cached textual weights W
    print("Getting cached textual weights W ...")
    feat_path = os.path.join(cfg['cache_dir'], f"{cfg['dataset']}_{cfg['backbone']}_textfeats.pt")
    text_feats = clip_classifier(feat_path, dataset.classnames, dataset.template, clip_model)

    # Preparation for training
    for param in clip_model.parameters():
        param.requires_grad = False
    for name, param in clip_model.named_parameters():
        if 'adapter' in name:
            param.requires_grad = True
    
    trainer = Trainer(cfg, clip_model, train_loader, test_loader, logger, text_feats, val_loader)
    trainer.train()


if __name__ == '__main__':
    main()