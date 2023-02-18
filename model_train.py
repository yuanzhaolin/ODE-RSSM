#!/usr/bin/python
# -*- coding:utf8 -*-

import torch
import os
import sys

import sys
from importlib import reload

from .dataset import prepare_training_dataset
from .train import train
from .lib import util
import hydra
from omegaconf import DictConfig, OmegaConf
import traceback
from .common import init_network_weights
import numpy as np


def set_random_seed(seed, logging):
    rand_seed = np.random.randint(0, 100000) if seed is None else seed
    logging('random seed = {}'.format(rand_seed))
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


def main_train(args, logging):
    # 设置随机种子，便于结果复现
    global scale
    set_random_seed(args.random_seed, logging)
    use_cuda = args.use_cuda and torch.cuda.is_available()

    # 根据args的配置生成模型
    from .model.generate_model import generate_model
    model = generate_model(args)

    # 模型加载到gpu上
    if use_cuda:
        model = model.cuda()

    if args.train.init_weights:
        init_network_weights(model)
    logging('save dir = {}'.format(os.getcwd()))
    logging(model)

    # 构建训练集和验证集
    train_loader, val_loader = prepare_training_dataset(args, logging)
    train(args, model, train_loader, val_loader, logging)


@hydra.main(config_path='config', config_name="config.yaml")
def main_app(args: DictConfig) -> None:
    from .common import SimpleLogger, training_loss_visualization

    logging = SimpleLogger('./log.out')

    # region loading the specific model configuration (config/paras/{dataset}/{model}.yaml)
    if args.use_model_dataset_config:
        model_dataset_config = util.load_DictConfig(
            os.path.join(hydra.utils.get_original_cwd(), 'config', 'paras', args.dataset.type),
            args.model.type + '.yaml'
        )
        if model_dataset_config is None:
            logging(f'Can not find model config file  in config/paras/{args.dataset.type}/{args.model.type}.yaml, '
                    f'loading default model config')
        else:
            args.model = model_dataset_config
    # endregion

    # In continuous-time mode, the last dimension of input variable is the delta of time step.
    if args.ct_time:
        args.dataset.input_size += 1

    # Save args for running model_test.py individually
    util.write_DictConfig('./', 'exp.yaml', args)

    logging(OmegaConf.to_yaml(args))

    # Model Training
    try:
        main_train(args, logging)
        training_loss_visualization('./')
    except Exception as e:
        var = traceback.format_exc()
        logging(var)

    # Evaluation in Test Dataset
    from .model_test import main_test
    ckpt_path = './'
    logging = SimpleLogger(
        os.path.join(
            ckpt_path, 'test.out'
        )
    )
    try:
        with torch.no_grad():
            main_test(args, logging, ckpt_path)
    except Exception as e:
        var = traceback.format_exc()
        logging(var)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_app()
