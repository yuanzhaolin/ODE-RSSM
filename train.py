#!/usr/bin/python
# -*- coding:utf8 -*-
import time
from .common import CTSample, vae_loss
from .common import RMSE, RRSE, get_eval_indice_dict
from .model.common import PeriodSchedule, ValStepSchedule
from .lib.util import TimeRecorder
import os
import torch
import numpy as np
from torch.utils.data import DataLoader


def eval_net(model, data_loader, epoch, args):
    acc_loss = 0
    acc_items = 0
    acc_rrse = 0
    acc_rmse = 0
    acc_time = 0
    acc_pred_likelihood = 0
    acc_multisample_rmse = 0

    model.eval()
    use_cuda = args.use_cuda and torch.cuda.is_available()

    tr = TimeRecorder()

    for i, data in enumerate(data_loader):

        external_input, observation = data
        external_input = external_input.permute(1, 0, 2)
        observation = observation.permute(1, 0, 2)
        if use_cuda:
            external_input = external_input.cuda()
            observation = observation.cuda()

        l, batch_size, _ = external_input.size()

        with tr('val'):
            # Update: 20210618 ，删掉训练阶段在model_train中调用forward_posterior的过程,直接调用call_loss(external_input, observation)
            losses = model.call_loss(external_input, observation)
            loss, kl_loss, likelihood_loss = losses['loss'], losses['kl_loss'], losses['likelihood_loss']

            if kl_loss != 0:
                loss = vae_loss(kl_loss, likelihood_loss, epoch, kl_inc=args.train.kl_inc,
                                kl_wait=args.train.kl_wait, kl_max=args.train.kl_max)

            # region Prediction
            prefix_length = max(int(args.dataset.history_length * args.sp),
                                1) if args.ct_time else args.dataset.history_length
            _, memory_state = model.forward_posterior(
                external_input[:prefix_length], observation[:prefix_length]
            )
            outputs, memory_state = model.forward_prediction(
                external_input[prefix_length:], n_traj=args.test.n_traj, memory_state=memory_state
            )

        acc_time += tr['val']

        acc_loss += float(loss) * external_input.shape[1]
        # 预测的likelihood
        pred_likelihood = - float(torch.sum(outputs['predicted_dist'].log_prob(observation[prefix_length:]))) / batch_size / (l - prefix_length)
        acc_pred_likelihood += pred_likelihood * external_input.shape[1]
        acc_items += external_input.shape[1]

        acc_rrse += float(RRSE(
            observation[prefix_length:], outputs['predicted_seq'])
        ) * external_input.shape[1]

        acc_rmse += float(RMSE(
            observation[prefix_length:], outputs['predicted_seq'])
        ) * external_input.shape[1]

        acc_multisample_rmse += np.mean([float(RMSE(
            observation[prefix_length:], outputs['predicted_seq_sample'][:, :, i, :])) for i in range(outputs['predicted_seq_sample'].shape[2])]
        ) * external_input.shape[1]

    model.train()
    return acc_loss / acc_items, acc_rrse / acc_items, acc_rmse/acc_items, acc_time / acc_items, \
           acc_pred_likelihood / acc_items, acc_multisample_rmse / acc_items


def train(args, model, train_dataset, val_dataset, logging):

    # region [data loader]
    collate_fn = None if not args.ct_time else CTSample(args.sp, args.base_tp, evenly=args.sp_even).batch_collate_fn
    train_loader = DataLoader(train_dataset, batch_size=args.train.batch_size,
                              shuffle=True, num_workers=args.train.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.train.batch_size, shuffle=False,
                            num_workers=args.train.num_workers, collate_fn=collate_fn)
    # endregion

    best_val = 1e12
    device = iter(model.parameters()).__next__().device
    use_cuda = False if device == torch.device('cpu') else True

    # 设置模型训练优化器
    if args.train.optim.type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.train.optim.lr)
    elif args.train.optim.type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.train.optim.lr)
    else:
        raise NotImplementedError

    # 学习率调整器
    if args.train.schedule.type == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.train.schedule.gamma)
    elif args.train.schedule.type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.train.schedule.step_size, gamma=args.train.schedule.gamma)
    elif args.train.schedule.type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train.schedule.T_max,
                                                               eta_min=args.train.schedule.eta_min)
    elif args.train.schedule.type == 'val_step':
        scheduler = ValStepSchedule(optimizer,
                                    args.train.schedule.lr_scheduler_nstart,
                                    args.train.schedule.lr_scheduler_nepochs,
                                    args.train.schedule.lr_scheduler_factor,
                                    logging)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=1)
        logging('No scheduler used in training !!!!')

    if hasattr(args.train.schedule, 'step_size'):
        scheduler = PeriodSchedule(scheduler, args.train.schedule.step_size, logging)

    logging('make train loader successfully. Length of loader: %i' % len(train_loader))

    best_dev_epoch = -1

    # all_val_rrse = []
    all_val_rmse = []
    model.train()
    # 开始训练，重复执行args.train.epochs次
    for epoch in range(args.train.epochs):
        acc_loss = 0
        acc_kl_loss = 0
        acc_likelihood_loss = 0
        acc_items = 0
        for i, data in enumerate(train_loader):
            t1 = time.time()

            external_input, observation = data
            acc_items += external_input.shape[0]

            external_input = external_input.permute(1, 0, 2)
            observation = observation.permute(1, 0, 2)

            if use_cuda:
                external_input = external_input.cuda()
                observation = observation.cuda()

            t2 = time.time()

            losses = model.call_loss(external_input, observation)
            loss, kl_loss, likelihood_loss = losses['loss'], losses['kl_loss'], losses['likelihood_loss']
            if kl_loss != 0:
                loss = vae_loss(kl_loss, likelihood_loss, epoch, kl_inc=args.train.kl_inc,
                                kl_wait=args.train.kl_wait, kl_max=args.train.kl_max)

            t3 = time.time()

            acc_loss += float(loss) * external_input.shape[1]
            acc_kl_loss += float(kl_loss) * external_input.shape[1]
            acc_likelihood_loss += float(likelihood_loss) * external_input.shape[1]

            optimizer.zero_grad()  # 清理梯度
            loss.backward()  # loss反向传播
            optimizer.step()  # 参数优化
            logging(
                'epoch-round = {}-{} with loss = {:.4f} kl_loss = {:.4f}  likelihood_loss = {:.4f} '
                'prepare time {:.4f} forward time {:.4f}, forward percent{:.4f}%'.format(
                    epoch, i, float(loss), float(kl_loss), float(likelihood_loss), t2 - t1, t3 - t2,
                                                                                   100 * (t3 - t2) / (t3 - t1))
            )
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logging(
            'epoch = {} with train_loss = {:.4f} with kl_loss = {:.4f} with likelihood_loss = {:.4f} learning_rate = {:.6f}'.format(
                epoch, float(acc_loss / acc_items), float(acc_kl_loss / acc_items), float(acc_likelihood_loss / acc_items),
                lr
            ))
        if (epoch + 1) % args.train.eval_epochs == 0:
            with torch.no_grad():
                val_loss, val_rrse, val_rmse, val_time, val_pred_likelihood, val_multisample_rmse = eval_net(model,
                                                                                                             val_loader,
                                                                                                             epoch, args)
            logging(
                'eval epoch = {} with loss = {:.6f} rmse = {:.4f} rrse = {:.4f} val_time = {:.4f} val_pred_likelihood = {:.4f} val_multisample_rmse = {:.4f} learning_rate = {:.6f}'.format(
                    epoch, val_loss, val_rmse, val_rrse, val_time, val_pred_likelihood, val_multisample_rmse, lr)
            )
            all_val_rmse.append(val_rmse)
            scheduler.step(all_val_rmse, val_rmse)
            # TODO:目前评价标准为rmse，需要同时考虑rmse\likelihood\multisample_rmse? 如何比较好的同时以三者为评价标准？ # 用likehood好一些，越大越好
            # epoch_val = val_pred_likelihood
            eval_indice_dict = get_eval_indice_dict(
                val_loss, val_rrse, val_rmse, val_pred_likelihood, val_multisample_rmse
            )
            epoch_val = eval_indice_dict[f"{args.train.eval_indice}"]
            if best_val > epoch_val:
                best_val = epoch_val
                best_dev_epoch = epoch
                ckpt = dict()
                ckpt['model'] = model.state_dict()
                ckpt['epoch'] = epoch + 1
                #  ckpt['scale'] = scale    # 记录训练数据的均值和方差用于控制部分归一化和反归一化
                torch.save(ckpt, os.path.join('./', 'best.pth'))
                torch.save(model.to(torch.device('cpu')), os.path.join('./', 'control.pkl'))
                if use_cuda:
                    model = model.cuda()
                logging('Save ckpt at epoch = {}'.format(epoch))

            if epoch - best_dev_epoch > args.train.max_epochs_stop and epoch > args.train.min_epochs:
                logging('Early stopping at epoch = {}'.format(epoch))
                break

        # Update learning rate
        if not args.train.schedule.type == 'val_step':
            scheduler.step()  # 更新学习率

        # lr - Early stoping condition
        if lr < args.train.optim.min_lr:
            logging('lr is too low! Early stopping at epoch = {}'.format(epoch))
            break

    logging('Training finished')
