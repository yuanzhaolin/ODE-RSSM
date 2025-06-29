#!/usr/bin/python
# -*- coding:utf8 -*-
import math
import os
import json

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import traceback
from matplotlib import pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

from .lib import util
from .common import CTSample, normal_interval, Statistic, vae_loss, detect_download

from .dataset import (WesternDataset, WesternConcentrationDataset,
                      CstrDataset, WindingDataset, IBDataset, WesternDataset_1_4,
                      NLDataset, ActuatorDataset, BallbeamDataset, DriveDataset,
                      DryerDataset, GasFurnaceDataset, SarcosArmDataset,
                      Thickener_Simulation)
from .model.common import DiagMultivariateNormal as MultivariateNormal
from .model.generate_model import generate_model



def set_random_seed(seed):
    rand_seed = np.random.randint(0, 100000) if seed is None else seed
    print('random seed = {}'.format(rand_seed))
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


def main_test(args, logging, ckpt_path):
    set_random_seed(args.random_seed)
    figs_path = os.path.join(logging.dir, 'figs')
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)

    if args.test.plt_single:   # 单图绘制
        single_figs_path = os.path.join(figs_path, 'single_figs')
        if not os.path.exists(single_figs_path):
            os.makedirs(single_figs_path)

    # 创建独立画图的目录
    independent_test_path = os.path.join(figs_path, 'independent_test')
    if not os.path.exists(independent_test_path):
        os.makedirs(independent_test_path)

    model = generate_model(args)
    ckpt = torch.load(
        os.path.join(
            ckpt_path, 'best.pth'
        )
    )
    model.load_state_dict(ckpt['model'])
    if args.use_cuda:
        model = model.cuda()
    model.eval()
    logging(model)

    if args.dataset.type == 'cstr':
        dataset = CstrDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.test_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type == 'winding':
        dataset = WindingDataset(pd.read_csv(
            os.path.join(hydra.utils.get_original_cwd(), args.dataset.test_path)
        ), args.dataset.history_length + args.dataset.forward_length, step=args.dataset.dataset_window)

    elif args.dataset.type.startswith('southeast'):
        from .dataset import SoutheastThickener

        data = np.load(os.path.join(hydra.utils.get_original_cwd(), args.dataset.data_path), allow_pickle=True)
        dataset = SoutheastThickener(data,
                                     length=args.dataset.history_length + args.dataset.forward_length,
                                     step=args.dataset.dataset_window,
                                     dataset_type='test', io=args.dataset.io, smooth_alpha=args.dataset.smooth_alpha
                                     )
    else:
        raise NotImplementedError

    test_loader = DataLoader(dataset, batch_size=args.test.batch_size, shuffle=False,
                             num_workers=args.train.num_workers, drop_last=False,
                             collate_fn=CTSample(args.sp, args.base_tp, evenly=args.sp_even).batch_collate_fn if args.ct_time else None)

    logging('make test loader successfully. Length of loader: %i' % len(test_loader))
    acc_items = 0
    acc_name = ['vll', 'mll',  'ob_rrse'] + \
               ['ob_{}_rrse'.format(name) for name in args.dataset.target_names] + \
               ['ob_{}_pear'.format(name) for name in args.dataset.target_names] + \
               ['pred_rrse'] + \
               ['pred_{}_rrse'.format(name) for name in args.dataset.target_names] + \
               ['pred_{}_pear'.format(name) for name in args.dataset.target_names] + \
               ['pred_likelihood'] + \
               ['pred_rmse'] + \
               ['pred_{}_rmse'.format(name) for name in args.dataset.target_names] + \
               ['pred_multisample_rmse'] + \
               ['pred_{}_multisample_rmse'.format(name) for name in args.dataset.target_names] + \
               ['time', 'num']
    acc_info = np.zeros(len(acc_name))

    def single_data_generator(acc_info):
        for i, data in enumerate(test_loader):

            external_input, observation = data
            # if args.dataset.type == 'southeast':
            #     inverse_ex_input = scaler.inverse_transform_input(external_input)
            #     inverse_out = scaler.inverse_transform_output(observation)

            external_input = external_input.permute(1, 0, 2)
            observation = observation.permute(1, 0, 2)

            if args.use_cuda:
                external_input = external_input.cuda()
                observation = observation.cuda()

            with torch.no_grad():
                outputs, memory_state = model.forward_posterior(external_input, observation)

                decode_observations_dist = model.decode_observation(outputs, mode='dist')
                decode_observations = model.decode_observation(outputs, mode='sample')
                decode_observation_low, decode_observation_high = normal_interval(decode_observations_dist, 2)

                # region Prediction
                prefix_length = max(int(args.dataset.history_length * args.sp), 1) if args.ct_time else args.dataset.history_length
                _, memory_state = model.forward_posterior(
                    external_input[:prefix_length], observation[:prefix_length]
                )
                outputs, memory_state = model.forward_prediction(
                    external_input[prefix_length:], args.test.n_traj, memory_state=memory_state
                )
            pred_observations_dist = outputs['predicted_dist']
            pred_observations_sample = outputs['predicted_seq']
            pred_observations_sample_traj = outputs['predicted_seq_sample']
            if args.model.type == 'vaecl':
                weight_map = memory_state['weight_map']
            else:
                weight_map = torch.zeros((2, 1, 2))  # minimal shape
            # endregion

            pred_observation_low, pred_observation_high = normal_interval(pred_observations_dist, 2)

            # region statistic
            variable_data_list = [
                observation,
                pred_observations_dist,
                pred_observations_sample,
                pred_observations_sample_traj,
                decode_observations,
                prefix_length
            ]
            losses = model.call_loss(external_input, observation)
            loss, kl_loss, likelihood_loss = losses['loss'], losses['kl_loss'], losses['likelihood_loss']

            # tensor2cpu
            tensor2cpu = lambda a: a.cpu() if type(a) == torch.Tensor else a
            loss = tensor2cpu(loss)
            kl_loss = tensor2cpu(kl_loss)
            likelihood_loss = tensor2cpu(likelihood_loss)

            if kl_loss != 0:
                loss = vae_loss(kl_loss, likelihood_loss, 1000, kl_inc=args.train.kl_inc,
                                kl_wait=args.train.kl_wait, kl_max=args.train.kl_max)

            # TODO:指标要加在这里
            ob_rrse, ob_rrse_single, ob_pear, prediction_rrse, prediction_rrse_single,\
            prediction_pearsonr, pred_likelihood, prediction_rmse, prediction_rmse_single, \
            pred_multisample_rmse, pred_multisample_rmse_single, time = Statistic(variable_data_list, split=False)

            ob_pearson_info = ' '.join(
                ['ob_{}_pear={:.4f}'.format(name, pear) for pear, name in zip(ob_pear, args.dataset.target_names)])
            pred_pearson_info = ' '.join(['pred_{}_pear={:.4f}'.format(name, pear) for pear, name in zip(
                prediction_pearsonr, args.dataset.target_names)])

            ob_rrse_info = ' '.join(
                ['ob_{}_rrse={:.4f}'.format(name, rrse) for rrse, name in
                 zip(ob_rrse_single, args.dataset.target_names)])
            pred_rrse_info = ' '.join(['pred_{}_rrse={:.4f}'.format(name, rrse) for rrse, name in zip(
                prediction_rrse_single, args.dataset.target_names)])
            pred_rmse_info = ' '.join(['pred_{}_rmse={:.4f}'.format(name, rmse) for rmse, name in zip(
                prediction_rmse_single, args.dataset.target_names)])
            # pred_likelihood_info = ' '.join(['pred_{}_likelihood={:.4f}'.format(name, likelihood) for likelihood, name in zip(
            #     pred_likelihood, args.dataset.target_names)])
            pred_multisample_rmse_info = ' '.join(['pred_{}_multisample_rmse={:.4f}'.format(name, multisample_rmse) for multisample_rmse, name in zip(
                pred_multisample_rmse_single, args.dataset.target_names)])

            log_str = 'seq = {} vll = {:.4f} mll={:.4f} ob_rrse={:.4f} ' + ob_rrse_info + ' ' + ob_pearson_info + \
                      ' pred_rrse={:.4f} ' + pred_rrse_info + ' ' + pred_pearson_info + \
                      ' pred_likelihood={:.4f} ' + \
                      ' pred_rmse={:.4f} ' + pred_rmse_info + \
                      ' pred_multisample_rmse={:.4f} ' + pred_multisample_rmse_info + 'time={:.4f}'
            logging(log_str.format(i, loss, likelihood_loss,
                                   ob_rrse,
                                   prediction_rrse,
                                   pred_likelihood,
                                   prediction_rmse,
                                   pred_multisample_rmse,
                                   time))

            acc_info += np.array([
                loss, likelihood_loss, ob_rrse, *ob_rrse_single, *ob_pear,
                prediction_rrse, *prediction_rrse_single, *prediction_pearsonr,
                pred_likelihood,
                prediction_rmse, *prediction_rmse_single,
                pred_multisample_rmse, *pred_multisample_rmse_single, time, 1
            ], dtype=np.float32) * external_input.size(1)

            for i in range(external_input.size(1)):
                yield tuple([x[:, i:i + 1, :] for x in [observation, decode_observations,
                                                        decode_observation_low, decode_observation_high,
                                                        pred_observation_low, pred_observation_high,
                                                        pred_observations_sample, pred_observations_sample_traj, external_input]] + [weight_map])

    for i, result in enumerate(single_data_generator(acc_info)):
        if i % int((len(dataset)-1) // args.test.plt_cnt + 1) == 0:

            observation, decode_observations, decode_observation_low, decode_observation_high, \
            pred_observation_low, pred_observation_high, pred_observations_sample, pred_observations_sample_traj, external_input = [x for x in
                                                                                     result[:-1]]
            # DiagMultivariateNormal类型的pred_observations_dist无法通过[x]转递
            pred_observations_dist = MultivariateNormal(
                pred_observations_sample_traj.mean(dim=2), torch.diag_embed(pred_observations_sample_traj.var(dim=2))
            )
            # 计算单条数据的预测指标
            prefix_length = observation.size(0) - pred_observations_sample.size(0)
            variable_data_list = [
                observation,
                pred_observations_dist,
                pred_observations_sample,
                pred_observations_sample_traj,
                decode_observations,
                prefix_length
            ]
            acc_info_single = Statistic(variable_data_list, split=True)


            # 遍历每一个被预测指标
            for _ in range(len(args.dataset.target_names)):
                observation, decode_observations, decode_observation_low, decode_observation_high, \
                pred_observation_low, pred_observation_high, pred_observations_sample, pred_observations_sample_traj = [x[..., _] for
                                                                                         x in
                                                                                         result[:-2]]
                external_input = result[-2]
                weight_map = result[-1]
                target_name = args.dataset.target_names[_]
                # region 开始画图
                plt.figure(figsize=(10, 8))
                ##################图一:隐变量区间展示###########################

                plt.subplot(221)
                # 单条数据不显示loss
                text_list = ['{}={:.4f}'.format(name, value) for name, value in
                             zip(acc_name[1:], acc_info_single)]
                for pos, text in zip(np.linspace(0, 1, len(text_list) + 1)[:-1], text_list):
                    plt.text(0.2, pos, text)
                # plt.plot(observation, label='observation')
                # plt.plot(estimate_state, label='estimate')
                # plt.fill_between(range(len(state)), interval_low, interval_high, facecolor='green', alpha=0.2,
                #                  label='hidden')
                # if args.dataset == 'fake':
                #     plt.plot(state, label='gt')
                #     #plt.plot(observation)

                # # southeast_ore_dateset适配
                # external_input = external_input.detach().cpu().squeeze()
                #
                # plt.plot(range(external_input.shape[0]), external_input, label=args.dataset.in_columns[0])
                # # plt.plot(range(external_input.shape[0]), external_input[:, 1])
                # plt.legend()

                x_all = range(len(observation))
                x_prefix = range(prefix_length)
                x_suffix = range(prefix_length, observation.size()[0])
                x_suffix_plus = range(prefix_length - 1, observation.size()[0])
                if args.ct_time:
                    x_all = torch.cumsum(external_input[x_all, 0, -1], dim=0).cpu().numpy() / args.base_tp
                    x_prefix, x_suffix, x_suffix_plus = [x_all[x_inds] for x_inds in [x_prefix, x_suffix, x_suffix_plus]]


                ##################图二:生成观测数据展示###########################
                plt.subplot(222)
                observation = observation.detach().cpu().squeeze(dim=1)
                estimate_observation_low = decode_observation_low.cpu().squeeze().detach()
                estimate_observation_high = decode_observation_high.cpu().squeeze().detach()
                plt.plot(x_all, observation, label=target_name, zorder=2)
                plt.fill_between(x_all, estimate_observation_low,
                                 estimate_observation_high,
                                 facecolor='green', alpha=0.2, label='95%', zorder=1)
                if args.ct_time:
                    plt.scatter(x_all, observation, s=1, c='black', zorder=3)
                plt.legend()
                if args.dataset.type == 'nl':
                    x_limit_show = [0, 200]
                    plt.xlim(x_limit_show)

                ##################图三:预测效果###########################
                plt.subplot(223)

                # observation = scaler.inverse_transform_output(observation)
                # pred_observation_low = scaler.inverse_transform_output(pred_observation_low)
                # pred_observation_high = scaler.inverse_transform_output(pred_observation_high)
                # pred_observations_sample = scaler.inverse_transform_output(pred_observations_sample)
                plt.plot(x_prefix, observation[:prefix_length], label='history')
                plt.plot(x_suffix_plus, observation[prefix_length - 1:], label='real', zorder=4)
                pred_observations_sample_traj = pred_observations_sample_traj.permute(2, 0, 1)
                pred_observations_sample_traj = pred_observations_sample_traj.detach().squeeze(dim=-1).cpu().numpy()
                plt_sample_cnt = min(args.test.plt_n_traj, pred_observations_sample_traj.shape[0])
                for n in range(plt_sample_cnt):
                    plt.plot(x_suffix_plus,
                             np.concatenate([[float(observation[prefix_length - 1])],
                                             pred_observations_sample_traj[n]]),
                             label=('prediction_sample' if n == 0 else None), color='grey', linewidth=0.3, alpha=0.9, zorder=2)
                plt.plot(x_suffix_plus,
                         np.concatenate([[float(observation[prefix_length - 1])],
                                         pred_observations_sample.detach().squeeze().cpu().numpy()]),
                         label='prediction', color='red', alpha=0.7, zorder=3)
                plt.fill_between(x_suffix,
                                 pred_observation_low.detach().squeeze().cpu().numpy(),
                                 pred_observation_high.detach().squeeze().cpu().numpy(),
                                 facecolor='lightgreen', alpha=0.4, label='95%',  zorder=1)
                if args.ct_time:
                    plt.scatter(x_suffix, pred_observations_sample.detach().squeeze().cpu().numpy(), s=1, c='black', zorder=3)
                plt.ylabel(target_name)
                plt.legend()

                if args.dataset.type == 'nl':
                    x_limit_show = [0, 200]
                    plt.xlim(x_limit_show)

                ##################图四:weight map 热力图###########################
                plt.subplot(224)
                weight = weight_map.mean(dim=1)  # 沿着batch维度求平均
                weight = weight.transpose(1, 0)
                weight = weight.detach().cpu().numpy()
                # cs = plt.contourf(weight, cmap=plt.cm.hot)
                cs = plt.contourf(weight)
                cs.changed()
                plt.colorbar()
                plt.xlabel('Steps')
                plt.ylabel('Weight of linears')

                # endregion 画图结束
                plt.savefig(
                    os.path.join(
                        figs_path, str(i) + '_' + str(_) + '.png'
                    )
                )
                plt.close()

                # 画单独的模型预测图，而不是在总图中只占用1/4
                if args.test.plt_single:
                    plt.figure(figsize=(7, 5))

                    plt.plot(x_prefix, observation[:prefix_length], label='history')
                    plt.plot(x_suffix_plus, observation[prefix_length - 1:], label='real', zorder=4)
                    # for n in range(int(pred_observations_sample_traj.shape[0])):
                    #     if n == 0:
                    #         plt.plot(x_suffix_plus,
                    #                  np.concatenate([[float(observation[prefix_length - 1])],
                    #                                  pred_observations_sample_traj[n]]),
                    #                  label='prediction_sample', color='grey', linewidth=0.3, alpha=0.9, zorder=2)
                    #     else:
                    #         plt.plot(x_suffix_plus,
                    #                  np.concatenate([[float(observation[prefix_length - 1])],
                    #                                  pred_observations_sample_traj[n]]),
                    #                  color='grey', linewidth=0.3, alpha=0.9, zorder=2)
                    plt.plot(x_suffix_plus,
                             np.concatenate([[float(observation[prefix_length - 1])],
                                             pred_observations_sample.detach().squeeze().cpu().numpy()]),
                             label='prediction', color='red', alpha=0.7, zorder=3)
                    plt.fill_between(x_suffix,
                                     pred_observation_low.detach().squeeze().cpu().numpy(),
                                     pred_observation_high.detach().squeeze().cpu().numpy(),
                                     facecolor='lightgreen', alpha=0.4, label='95%', zorder=1)
                    if args.ct_time:
                        plt.scatter(x_suffix, pred_observations_sample.detach().squeeze().cpu().numpy(), s=1, c='black',
                                    zorder=3)
                    plt.ylabel(target_name)
                    plt.legend()

                    # plt.savefig(
                    #     os.path.join(
                    #         single_figs_path, str(i) + '_' + str(_) + '.eps'
                    #     ), dpi=600
                    # )
                    plt.savefig(
                        os.path.join(
                            single_figs_path, str(i) + '_' + str(_) + '.pdf'
                        ), dpi=600
                    )
                    plt.savefig(
                        os.path.join(
                            single_figs_path, str(i) + '_' + str(_) + '.png'
                        )
                    )
                    plt.close()

                    def to_list(x):
                        if isinstance(x, list):
                            return x
                        try:
                            return x.cpu().numpy().tolist()
                        except Exception as e:
                            pass
                        return x.tolist()

                    with open(os.path.join(single_figs_path, str(i) + '_' + str(_) + '.json'), 'w') as f:
                        json.dump(
                            {
                                'x_prefix': to_list(x_prefix),
                                'observation': to_list(observation),
                                'prefix_length': prefix_length,
                                'x_suffix_plus': to_list(x_suffix_plus),
                                'pred_observations_sample': to_list(pred_observations_sample),
                                'x_suffix': to_list(x_suffix),
                                'pred_observation_low': to_list(pred_observation_low),
                                'pred_observation_high': to_list(pred_observation_high),
                                'target_name': target_name,
                            }, f
                        )

    logging(' '.join(
        ['{}={:.4f}'.format(name, value / acc_info[-1] if name != 'num' else 1) for name, value in zip(acc_name, acc_info)]
    ))


@hydra.main(config_path='config', config_name="config.yaml")
def main_app(args: DictConfig) -> None:

    from .common import SimpleLogger

    if not hasattr(args.test, 'test_dir') or args.test.test_dir is None:
        raise AttributeError('It should specify save_dir attribute in test mode!')

    # ckpt_path = '/code/SE-VAE/ckpt/southeast/seq2seq/seq2seq_ctrl_solution=2/2021-06-12_23-06-08'
    # ckpt_path = '/code/SE-VAE/ckpt/southeast/tmp/vaecl_/2021-06-18_20-52-31'
    ckpt_path = args.test.test_dir

    logging = SimpleLogger(os.path.join(ckpt_path, 'test.out'))

    # region load the config of original model
    exp_config = util.load_DictConfig(
        ckpt_path,
        'exp.yaml'
    )
    if exp_config is not None:
        exp_config.test = args.test
        if args.test.sp_change:
            exp_config.sp = args.sp
            exp_config.sp_even = args.sp_even
        args = exp_config

    # endregion
    logging(OmegaConf.to_yaml(args))

    try:
        with torch.no_grad():
            main_test(args, logging, ckpt_path)
    except Exception as e:
        var = traceback.format_exc()
        logging(var)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_app()
