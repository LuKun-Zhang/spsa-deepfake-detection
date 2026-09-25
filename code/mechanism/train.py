# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: training code.

import os
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import *
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter


parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='/data/home/zhiyuanyan/DeepfakeBenchv2/training/config/detector/sbi.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--task_target', type=str, default="", help='specify the target of current training task')
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    np.random.seed(config['manualSeed'])  # seedfix: np.random 此前不受 manualSeed 控制
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])
    # seedfix2: GPU 确定性（cuBLAS 归约/卷积算法选择），配合启动命令 CUBLAS_WORKSPACE_CONFIG=:4096:8
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id, dataset=None):
    # seedfix: DataLoader worker 内 seed，保证采样/增强随机性跨运行可复现
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    if dataset is not None:
        tr = getattr(dataset, 'transform', None)
        if tr is not None and hasattr(tr, 'set_random_seed'):
            tr.set_random_seed(worker_seed)  # albumentations compose 实例 RNG 从系统熵初始化，必须显式 seed


def prepare_training_data(config):
    # Only use the blending dataset class in training
    if 'dataset_type' in config and config['dataset_type'] == 'blend':
        if config['model_name'] == 'facexray':
            train_set = FFBlendDataset(config)
        elif config['model_name'] == 'fwa':
            train_set = FWABlendDataset(config)
        elif config['model_name'] == 'sbi':
            train_set = SBIDataset(config, mode='train')
        elif config['model_name'] == 'lsda':
            train_set = LSDADataset(config, mode='train')
        else:
            raise NotImplementedError(
                'Only facexray, fwa, sbi, and lsda are currently supported for blending dataset'
            )
    elif 'dataset_type' in config and config['dataset_type'] == 'pair':
        train_set = pairDataset(config, mode='train')  # Only use the pair dataset class in training
    elif 'dataset_type' in config and config['dataset_type'] == 'iid':
        train_set = IIDDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'I2G':
        train_set = I2GDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'lrl':
        train_set = LRLDataset(config, mode='train')
    else:
        train_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='train',
                )
    if config['model_name'] == 'lsda':
        from dataset.lsda_dataset import CustomSampler
        custom_sampler = CustomSampler(num_groups=2*360, n_frame_per_vid=config['frame_num']['train'], batch_size=config['train_batchSize'], videos_per_group=5)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                sampler=custom_sampler, 
                collate_fn=train_set.collate_fn,
            )
    elif config['ddp']:
        sampler = DistributedSampler(train_set)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                sampler=sampler
            )
    else:
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                worker_init_fn=lambda wid: seed_worker(wid, train_set),
                collate_fn=train_set.collate_fn,
                )
    return train_data_loader


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        if not config.get('dataset_type', None) == 'lrl':
            test_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='test',
            )
        else:
            test_set = LRLDataset(
                config=config,
                mode='test',
            )

        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                worker_init_fn=lambda wid: seed_worker(wid, test_set),
                collate_fn=test_set.collate_fn,
                drop_last = (test_name=='DeepFakeDetection'),
            )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(), 
            optim.SGD, 
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['nEpochs'],
            int(config['nEpochs']/4),
        )
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    # 修复：detector 特有字段（SWA 等）优先，避免被 train_config.yaml 的默认值覆盖
    # 2026-09-16 机制实验扩表：log_dir 尤其关键——train_config.yaml 把它钉在
    # ./logs/training/（数据盘，94% 满），而机制实验的 24 腿产物约 21 GB，必须落到
    # 系统盘 /root/swa_exp；dry_run 同理（全局为 false，冒烟测试要从 yaml 打开）。
    for _k in ['SWA', 'swa_start', 'swa_start_lr_ratio', 'lr_scheduler', 'lr_T_max', 'lr_eta_min',
               'log_dir', 'dry_run', 'cudnn', 'manualSeed', 'nEpochs', 'save_ckpt',
               'resolution', 'spsa', 'spsa_mode', 'tail_lam', 'rng_neutral_build',
               'swa_ws', 'test_times_per_epoch', 'probe_every', 'probe_grad_every',
               'bn_recalib_batches',
               # 2026-09-17 PHASE 1 噪声操纵：窗口内各向同性权重噪声的全局范数目标
               # 与它的专用 RNG 种子。不在这张表里的键会被 train_config.yaml 的
               # 默认值覆盖（合并方向是 detector yaml 优先，但白名单是门槛）。
               'window_noise_global', 'window_noise_seed']:
        if _k in config:
            config2[_k] = config[_k]
    config.update(config2)
    config['local_rank']=args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    # create logger
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = f"_{config['task_target']}" if config.get('task_target', None) is not None else ""
    logger_path =  os.path.join(
                config['log_dir'],
                config['model_name'] + task_str + '_' + timenow
            )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    config['ddp']= args.ddp
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        # dist.init_process_group(backend='gloo')
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=30)
        )
        logger.addFilter(RankFilter(0))
    # prepare the training data loader
    train_data_loader = prepare_training_data(config)

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    # --- RNG-neutral model construction (mechanism experiment) --------------
    # Building the detector consumes global RNG for Conv2d init: SPSA makes 5
    # convs, SPSAPlaceholder 3, CTRL 0.  That happens BEFORE the DataLoader
    # draws the seeds that drive shuffling and augmentation, so arms with
    # different modules would train on DIFFERENT data orders -- confounding
    # "has the module" with "got a different shuffle".
    # Measured on 871 (seed 2048): the three arms drew base seeds
    #   3302591561589991144 (ctrl) / 3547969457712062677 (ph) / 419313495234897532 (spsa)
    # and completely different first randperm(10).  Save/restore around the
    # construction makes the comparison clean.  Flag off => old behaviour.
    if config.get('rng_neutral_build', False):
        _cpu_rng = torch.get_rng_state()
        _cuda_rng = (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None)
    model = model_class(config)
    if config.get('rng_neutral_build', False):
        torch.set_rng_state(_cpu_rng)
        if _cuda_rng is not None:
            torch.cuda.set_rng_state_all(_cuda_rng)

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    # prepare the swa model (SWA: stochastic weight averaging)
    # `swa_ws` = window WIDTHS, as fractions of the collection window.  All of
    # them share the same endpoint and differ only in how far back they start,
    # so one run yields the whole dose-response curve AUC(theta_bar(w)).
    # AveragedModel is a deepcopy and draws no random numbers, so building five
    # of them instead of one does NOT perturb the training RNG stream -- which
    # the rng_neutral_build patch above depends on.  Unset => [1.0] == old behaviour.
    swa_model = None
    swa_models = {}
    if config.get('SWA', False):
        for _w in (config.get('swa_ws', None) or [1.0]):
            swa_models[float(_w)] = AveragedModel(model).cuda()
        swa_model = swa_models.get(1.0, next(iter(swa_models.values())))
        logger.info('SWA window widths: %s' % ','.join('%g' % _w for _w in sorted(swa_models)))

    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, time_now=timenow,
                      swa_model=swa_model, swa_models=swa_models)

    # start training
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
                    epoch=epoch,
                    train_data_loader=train_data_loader,
                    test_data_loaders=test_data_loaders,
                )
        if best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!")
        if scheduler is not None:
            lr_now = scheduler.get_last_lr()[0]
            logger.info(f"===> Epoch[{epoch}] end, current lr = {lr_now:.6f}")
            scheduler.step()
    logger.info("Stop Training on best Testing metric {}".format(parse_metric_for_print(best_metric))) 
    # 窗口末的损失面探针（设计 §5.2 / 理论 定理1·3·5）：J(w)、弦剖面、势垒 B、
    # DriftGap。必须在这里做 —— 再往后 trainer.model 会被换成 θ̄(w)，θ_end 就没了。
    if config.get('SWA', False) and trainer.swa_model is not None:
        trainer._window_end_probe()
    # 最终测试：用 SWA 平均模型（每个窗口宽度 w 各测一次）
    if config.get('SWA', False) and trainer.swa_model is not None:
        _ws = sorted(trainer.swa_models)
        for _w in _ws:
            trainer.save_swa_ckpt(w=_w)
        # ascending so that w=1.0 is evaluated last and `trainer.model` ends up
        # where it did in the single-model version
        for _w in _ws:
            trainer.model = trainer.swa_models[_w]
            trainer.model.epoch = config['nEpochs'] + 1
            trainer._swa_eval_w = _w
            trainer.test_epoch(config['nEpochs'] + 1, -1, test_data_loaders, 0)
            logger.info("===> Final SWA testing done! (w=%g)" % _w)
        # The same five, with BatchNorm running stats recomputed for each
        # theta_bar.  AveragedModel copies the buffers from the model it averages
        # on every update, so all five widths end up carrying theta_end's
        # statistics -- exactly right only while the window does not move; the
        # collected evidence (see trainer._bn_recalibrate) says the protocol's
        # deployment of a wide average is numerically destroyed at a window lr of
        # 2e-4.  Reporting both columns is what lets the sign of G be read as a
        # result rather than an artefact.
        trainer._swa_eval_bn = True
        for _w in _ws:
            _m = trainer.bn_model(_w)
            if _m is None:
                logger.warning("BN recalibration unavailable for w=%g" % _w)
                continue
            trainer.model = _m
            trainer.model.epoch = config['nEpochs'] + 1
            trainer._swa_eval_w = _w
            trainer.test_epoch(config['nEpochs'] + 1, -1, test_data_loaders, 0)
            logger.info("===> Final SWA testing done! (w=%g, BN recalibrated)" % _w)
        trainer._swa_eval_bn = False
        trainer.model = trainer.swa_models[_ws[-1]]
    # 最终测试：用 EMA 平均权重（若启用）
    if config.get('EMA', False) and getattr(trainer, 'use_ema', False):
        trainer.model.load_state_dict(trainer._ema_sd())
        trainer.model.epoch = config['nEpochs'] + 1
        trainer.test_epoch(config['nEpochs'] + 1, -1, test_data_loaders, 0)
        logger.info("===> Final EMA testing done!")
    # 最终测试：用 Lookahead slow 权重（若启用，Lookahead 的部署物）
    if config.get('Lookahead', False) and getattr(trainer, 'use_la', False):
        torch.save(trainer._la_slow_sd(), os.path.join(trainer.log_dir, 'la_slow.pth'))
        trainer.model.load_state_dict(trainer._la_slow_sd())
        trainer.model.epoch = config['nEpochs'] + 1
        trainer.test_epoch(config['nEpochs'] + 1, -1, test_data_loaders, 0)
        logger.info("===> Final Lookahead testing done!")
    # update
    if 'svdd' in config['model_name']:
        model.update_R(epoch)

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()



if __name__ == '__main__':
    main()
