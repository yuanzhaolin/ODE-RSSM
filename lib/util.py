#!/usr/bin/python
# -*- coding:utf8 -*-
import yaml
import numpy as np
import time
import os
import re
import pandas as pd
import itertools

from pandas import DataFrame
from omegaconf import DictConfig, OmegaConf
from functools import reduce
from operator import and_, or_


def load_DictConfig(path, name):
    file = os.path.join(
        path, name
    )
    if not os.path.exists(os.path.join(path, name)):
        return None

    config = OmegaConf.load(file)

    return DictConfig(config)


def write_DictConfig(path, name, exp_dict: DictConfig):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, name), "w", encoding="utf-8") as f:
        OmegaConf.save(exp_dict, f)


def load_yaml(path, name):

    file = os.path.join(
        path, name
    )
    if not os.path.exists(os.path.join(path, name)):
        return None

    with open(file, "r", encoding="utf-8") as f:
        config = yaml.load(f)

    return DictConfig(config)


class TimeRecorder:
    def __init__(self):
        self.infos = {}

    def __call__(self, info, *args, **kwargs):
        class Context:
            def __init__(self, recoder, info):
                self.recoder = recoder
                self.begin_time = None
                self.info = info

            def __enter__(self):
                self.begin_time = time.time()

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.recoder.infos[self.info] = time.time() - self.begin_time

        return Context(self, info)

    def __str__(self):
        return ' '.join(['{}:{:.2f}s'.format(info, t) for info, t in self.infos.items()])

    def __getitem__(self, item):
        return self.infos[item]


def my_filter(path):
    log_path = os.path.join('..', path, 'log.out')
    s = open(log_path, 'r').readlines()
    log_path = os.path.join('..', path, 'test.out')
    test_s = open(log_path, 'r').readlines()
    if False:
        return False
    elif len(test_s) < 5:
        return False
    elif 'Error' in test_s[-2]:
        return False
    else:
        return True


def generating_dir(base_dir, root_dir):
    ret = []
    for file in os.listdir(os.path.join(root_dir, base_dir)):
        if file == 'tmp':
            continue
        path = os.path.join(base_dir, file)
        try:
            if os.path.isdir(os.path.join(root_dir, path)):
                ret = ret + generating_dir(path, root_dir)
            elif file == 'test.out' and os.path.exists(os.path.join(root_dir, base_dir, 'best.pth')):
                # print(path)
                if my_filter(os.path.split(path)[0]):
                    ret.append(os.path.split(path)[0])
                else:
                    continue
        except Exception as e:
            print(f'Generating dir {path} failed')
            continue

    return ret


def generate_data_frame(ckpt_dir='ckpt', root_dir='../', date=None, model=None, key_words=None,
                        dataset=None, sort_key='pred_rmse'):
    """

    Args:
        date: only counting the logs later than date
        model:  save_dir
        ckpt_dir: the name of ckpt dir
        root_dir: the position of ckpt_dir

    Returns:
        data_frames: List of Dataframe. {Dataframe-1, Dataframe-2, ... }
        datasets: List of str. {datset-1, datset-2 ,...}
        path_list: List of str, all paths of ckpt

    How to use:

        datasets, data_frames, path_list = generate_data_frame(ckpt_dir='ckpt', root_dir='../', sort_key='likelihood',
            dataset='winding', sort_key='pred_rrse', key_words=['rssm', 'vrnn'])
    """
    if key_words is None:
        key_words=['/']
    if isinstance(key_words, str):
        key_words = [key_words]

    if not isinstance(key_words[0], list):
        key_words = [key_words]

    if isinstance(model, str):
        model = [model]

    path_list = generating_dir(ckpt_dir, root_dir)
    date_filter = lambda path: True if date is None else path.split('/')[-1] >= date
    model_filter = lambda path: True if model is None else (path.split('/')[2] in model or path.split('/')[3] in model)
    dataset_filter = lambda path: True if dataset is None else path.split('/')[1] == dataset
    key_words_filter = lambda path: True if key_words is None else reduce(
        or_, [reduce(and_, [kw in path for kw in kws]) for kws in key_words]
    )

    path_list = list(filter(lambda x: reduce(and_, [
        date_filter(x),
        model_filter(x),
        dataset_filter(x),
        key_words_filter(x),
    ]), path_list))

    data = list(set([path.split('/')[1] for path in path_list]))
    # Get the dataset/model name and number
    print(data)
    # data = data[:-1]

    df = []  # Collection of DataFrames

    for d in range(len(data)):
        temp_list = []  # Temporary list holding a single dataset
        dex = []  # row list
        col = []  # column list

        for path in path_list:
            if path.split('/')[1] == data[d]:
                temp_list.append(path)

        for path in temp_list:
            result = path.split('/')
            dex.append('//'.join(result[3:]))
        n_dex = len(dex)  # 行数

        # Open a separate path file to get the number of columns for initializing the numerical matrix
        for _, dir_path in enumerate(temp_list):

            with open('../' + dir_path + '/test.out', 'r') as f:
                temp_data = f.readlines()
                if re.search('Error', ' '.join(temp_data[-5:])):
                    continue

            col2id = {}
            for line in temp_data:
                if re.search('Time_sec', line) and not re.search('seq', line):
                    t_col = re.findall(r'(\w*\(?\w\)?\w*)=(-?[0-9]\d*\.\d*)', line)
                    for i in range(len(t_col)):
                        if t_col[i][0] == 'time':
                            pass
                        else:
                            col.append(t_col[i][0])
                            col2id[t_col[i][0]] = len(col) - 1
            break

        col = col + ['seed']
        n_col = len(col)  # number of columns
        x = np.zeros((n_dex, n_col))  # Numerical matrix + random seed
        for t in range(len(temp_list)):  # path in the t-th temporary list
            try:
                f = open('../' + temp_list[t] + '/test.out', 'r')
                temp_data = f.readlines()
                f.close()
                # Numerical matrix generation
                for line in temp_data:
                    if line.startswith('random_seed'):
                        rand_seed = re.findall(r'random_seed: (.*)', line)[0]
                        x[t, -1] = None if str(rand_seed) == 'null' else int(rand_seed)

                    if re.search('Time_sec', line) and not re.search('seq', line):
                        pattern = re.compile(r'-?[0-9]\d*\.\d*')  # Find numbers (finds that contain a decimal point)
                        result = pattern.findall(line)
                        # Fill in the row with the corresponding data
                        t_x = re.findall(r'(\w*\(?\w\)?\w*)=(-?[0-9]\d*\.\d*)', line)
                        for i in range(len(t_x)):
                            if t_x[i][0] == 'time':
                                pass
                            else:
                                if t_x[i][0] in col2id.keys():
                                    x[t, col2id[t_x[i][0]]] = t_x[i][1]  # The statistical indicators of different ckpts may be inconsistent

            except Exception as e:
                print(temp_list[t], ' is not identificated')
                raise e

        # Generate DataFrame
        df.append(DataFrame(x, columns=col, index=dex).sort_values(by=sort_key))
        # The current data set frame is generated, and the temporary list is cleared.

    return data, df, path_list


def df2table(path, output_path=None):
    # construct a zero dataframe, columns = ['0.25','0.5','1'], indexes = ['RSSM','RSSM-O']
    df = DataFrame(
        index=['', 'VAE-RNN', 'SRNN', 'STORN', 'RSSM', 'RSSM-O', 'ODE-RNN', 'Time-Aware', 'ODE-RSSM', 'ODE-RSSM-O'],
        columns=['25%(uneven)','25%(uneven)-E','25%(even)','25%(even)','50%(uneven)','50%(uneven)','50%(even)','50%(even)','100%','100%']
    )
    df.iloc[0, :] = ['MLL', 'RMSE','MLL', 'RMSE','MLL', 'RMSE','MLL', 'RMSE','MLL', 'RMSE']
    original_data = pd.read_csv(path)

    sp_keys = [f'sp={_}' for _ in ['0.25', '0.5', '1']]
    even_keys = ['sp_even=.alse', 'sp_even=.rue'] # 适配true/false, True/False的大小写
    # Cartesian produc of sp_keys and even_keys
    # keys = [''.join(i) for i in product(sp_keys, even_keys)]

    index_mapping = {
        'VAE-RNN': lambda x: 'vae-rnn' in x or 'vae_rnn' in x,
        'SRNN': lambda x: 'srnn' in x,
        'STORN': lambda x: 'storn' in x,
        'ODE-RNN': lambda x: 'ode_rnn' in x or 'ode-rnn' in x,
        'Time-Aware': lambda x: 'time_aware' in x or 'time-aware' in x,
        'RSSM': lambda x: 'final_rssm' in x and 'model.D=1,' in x,
        'RSSM-O': lambda x: 'final_rssm' in x and 'model.D=1,' not in x,
        'ODE-RSSM': lambda x: 'ode_rssm' in x and 'model.D=1,' in x,
        'ODE-RSSM-O': lambda x: 'ode_rssm' in x and 'model.D=1,' not in x,
    }
    # print(original_data.index)
    for group_id, key in enumerate(itertools.product(sp_keys, even_keys)):
        sp_key, even_key = key
        if sp_key.endswith('1') and even_key.endswith('rue'):
            # When sp = 1, it does not distinguish between even=false or even=True
            continue
        # Finding the rows in original_data that both sp_key and even_key exist in index
        for row_id in range(len(original_data.index)):
            ckpt_name = original_data.iloc[row_id, 0]
            if sp_key in ckpt_name and \
                    (sp_key.endswith('1') or len(re.findall(even_key, ckpt_name)) != 0):
                # Entering here indicates that sp_key and even key match successfully
                likelihood_idx = group_id * 2
                rmse_idx = group_id * 2 + 1
                for idx, filter_map in index_mapping.items():
                    if filter_map(ckpt_name):
                        def update_value(a, b):
                            if np.isnan(a) or b < a:
                                return b
                            else:
                                return a
                        df.loc[idx][likelihood_idx] = update_value(df.loc[idx][likelihood_idx], float(original_data.iloc[row_id]['pred_likelihood']))
                        df.loc[idx][rmse_idx] = update_value(df.loc[idx][rmse_idx], float(original_data.iloc[row_id]['pred_multisample_rmse']))
    # 3 decimal places
    df = df.round(3)
    if output_path is None:
        output_path = os.path.join(path.split('/')[:-1], path.split('/')[-1].split('.')[0] + 'output' + '.xlsx')
    df.to_excel(output_path)
    df.to_csv(output_path.replace('.xlsx', '.csv'))


if __name__ == '__main__':
    df2table('./result_winding.csv', './result_winding_output.xlsx')
