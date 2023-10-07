import os
import argparse
import yaml

import torch
from torch.utils.data import DataLoader

import clip
from utils import *
from logger import *
from eval import Eval
from datasets.imagenet import MyImageNet
from datasets.imagenetv2 import ImageNetV2
from datasets.imagenet_sketch import ImageNetSketch
from datasets.utils import _transform


def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings in yaml format')
    
    parser.add_argument('--shots', default=16, type=int, help='number of shots for each class in training')
    parser.add_argument('--title', type=str, default='default_title', help='title of the trained model')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size in testing')
    parser.add_argument('--resume', default=None, type=str, help='path that model resumed from')
    parser.add_argument('--log_file', default='log', type=str, help='log file')
    parser.add_argument('--desc', default='default description', type=str, help='more details and description of this testing')
    
    parser.add_argument('--seed', default=1, type=int, help='seed for the whole training')
    
    args = parser.parse_args()
    
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    for arg in vars(args):
        cfg[arg] = getattr(args, arg)
    torch.set_num_threads(cfg['num_threads'])

    return cfg


def get_test_loader(cfg, dataset='imagenet'):
    dataloader_params = dict(batch_size=cfg['batch_size'], num_workers=8, shuffle=False)
    if dataset == 'imagenet':
        dataset_params =  dict(root=cfg['data_path'], num_shots=cfg['shots'], split='test', transform=(_transform(224, is_train=False), None))
        dataset = MyImageNet(**dataset_params)
    elif dataset == 'imagenetv2':
        path = '/data/Public/Datasets/imagenetv2/imagenetv2-matched-frequency-format-val/'
        dataset_params =  dict(root=path, transform=_transform(224, is_train=False))
        dataset = ImageNetV2(**dataset_params)
    else:
        path = '/data/Public/Datasets/imagenet-sketch/sketch/'
        dataset_params =  dict(root=path, transform=_transform(224, is_train=False))
        dataset = ImageNetSketch(**dataset_params)
        
    test_loader = DataLoader(dataset=dataset, **dataloader_params)
    return dataset, test_loader


def main():

    # Load config file
    cfg = get_arguments()
    setup_seed(cfg['seed'])
    
    # Load clip model
    clip_model, preprocess = clip.load(cfg['backbone'], download_root='./model/clip')
    clip_model.eval()
    
    # Get dataset and dataloader
    print("Preparing dataset and dataloader.")
    test_dataset = 'imagenetv2' # imagenet imagenetv2 imagenetsketch
    testset, test_loader = get_test_loader(cfg, test_dataset)
    
    # Show config
    assert cfg['num_classes'] == len(testset.classnames)
    print("\nRunning configs.")
    print(cfg, "\n")
    print(f"Resume from: {cfg['resume']}")
    
    # Initialize diretory and logger
    dir_name = f"s{cfg['shots']}_{cfg['dataset']}_{cfg['backbone']}_test"
    log_dir = f'./log/{dir_name}'
    make_dirs(log_dir)
    setup_logging(save_dir=log_dir, file_name=cfg['log_file'])
    logger = logging.getLogger(name=cfg['title'])
    # log_init(logger, cfg)

    # Load cached textual weights W
    print("Getting cached textual weights W ...")
    feat_path = os.path.join(cfg['cache_dir'], f"{cfg['dataset']}_{cfg['backbone']}_textfeats.pt")
    text_feats = clip_classifier(feat_path, testset.classnames, testset.template, clip_model)

    # Load model and evaluate
    clip_model = load_model(cfg['resume'], clip_model)[0]
    logger.info(f"Test for {test_dataset}")
    eval = Eval(cfg, clip_model, test_loader, text_feats, logger)
    eval.eval()    
    logger.info(f'-------------------- END TESTING --------------------')
           

if __name__ == '__main__':
    main()