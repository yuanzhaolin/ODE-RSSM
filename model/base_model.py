#!/usr/bin/python
# -*- coding:utf8 -*-
import torch


class BaseModel:

    def forward_posterior(self, external_input_seq, observations_seq, memory_state=None):
        return self._forward_posterior(external_input_seq, observations_seq, memory_state)

    def forward_prediction(self, external_input_seq, n_traj, memory_state=None, grad=False):
        if grad:
            return self._forward_prediction(external_input_seq, n_traj, memory_state=memory_state)
        else:
            with torch.no_grad():
                return self._forward_prediction(external_input_seq, n_traj, memory_state=memory_state)

    def _forward_posterior(self, external_input_seq, observations_seq, memory_state=None):
        """
       Args:
            external_input_seq: 输入序列 (len, batch_size, input_size)
            observations_seq: 观测序列 (len, batch_size, ob_size)
            memory_state: 字典，模型记忆

        Returns: 元组 (
            outputs: 随便定义，只要输入给decode_observation能够完成重构即可
            memory_state: 字典，模型记忆
            )

        需要满足:
            outputs, memory_state = self.forward_posterior(external_input_seq, observations_seq, memory_state)
            reconstruction_observations_seq = self.decode_observation(outputs, mode='sample')
        """

        raise NotImplementedError

    def _forward_prediction(self, external_input_seq, n_traj, memory_state=None):
        """
        Args:
            external_input_seq: 输入序列 (len, batch_size, input_size)
            n_traj: 采样次数
            max_prob: 如果为False，从预测分布中随机采样，如果为True ,返回概率密度最大的估计值
            memory_state: 字典，模型记忆

        Returns: 元祖 (
            outputs: 预测结果的字典， 至少包含predicted_seq_sample, predicted_dist, predicted_seq三个key
            memory_state: 字典，模型记忆
            )
        }
        """
        raise NotImplementedError


    def call_loss(self, external_input_seq, observations_seq, memory_state=None):
        """
        Args:

        Returns:
            三个torch的标量: loss, kl_loss, decoding_loss，对于非VAE模型，后两位返回0

        loss要在batch_size纬度上取平均

        """
        raise NotImplementedError

    def decode_observation(self, outputs, mode='sample'):
        """

        Args:
            mode: dist or sample

        Returns:
            model为sample时，从分布采样(len,batch_size,observation)
            为dist时，直接返回分布对象torch.distributions.MultivariateNormal

        方法调用时不会给额外的输入参数，需在每次forward_prediction和forward_posterior之后将解码所需的信息存储在self里
        """
        raise NotImplementedError
