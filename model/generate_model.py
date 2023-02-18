#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from . import VAERNN, ODERSSM, RSSM,  LatentSDE


def generate_model(args):
    # if args.model.type == 'vaecl':
    #     # model = vaeakf_cl.VAEAKFCombinedLinear(
    #     model = vaeakf_combinational_linears.VAEAKFCombinedLinear(
    #         input_size=args.dataset.input_size - (
    #             1 if args.dataset.type == 'southeast' and args.ctrl_solution == 2 else 0),
    #         state_size=args.model.state_size,
    #         observations_size=args.dataset.observation_size + (
    #             1 if args.dataset.type == 'southeast' and args.ctrl_solution == 2 else 0),
    #         k=args.model.posterior.k_size,
    #         num_layers=args.model.posterior.num_layers,
    #         L=args.model.L,
    #         R=args.model.R,
    #         D=args.model.D,
    #         num_linears=args.model.dynamic.num_linears
    #     )
    if args.model.type == 'rssm':
        model = RSSM(
            input_size=args.dataset.input_size,
            state_size=args.model.state_size,
            observations_size=args.dataset.observation_size,
            k=args.model.k_size,
            num_layers=args.model.num_layers,
            D=args.model.D
        )
    elif args.model.type == 'vaernn':
        model = VAERNN(
            input_size=args.dataset.input_size,
            state_size=args.model.state_size,
            observations_size=args.dataset.observation_size,
            k=args.model.k_size,
            num_layers=args.model.num_layers
        )
    elif args.model.type == 'ode_rssm':
        assert args.ct_time
        model = ODERSSM(
            input_size=args.dataset.input_size,
            state_size=args.model.state_size,
            observations_size=args.dataset.observation_size,
            ode_num_layers=args.model.ode_num_layers,
            k=args.model.k_size,
            D=args.model.D,
            ode_hidden_dim=args.model.ode_hidden_dim,
            ode_solver=args.model.ode_solver,
            rtol=args.model.rtol,
            atol=args.model.atol,
            ode_type=args.model.ode_type,
            weight=args.model.weight,
            detach=args.model.detach,
            ode_ratio=args.model.ode_ratio,
            iw_trajs=args.model.iw_trajs,
            z_in_ode=args.model.z_in_ode,
            input_interpolation=args.model.input_interpolation,
        )
    elif args.model.type == 'latent_sde':
        assert args.ct_time
        model = LatentSDE(
            h_size=args.model.h_size,
            u_size=args.dataset.input_size,
            y_size=args.dataset.observation_size,
            theta=args.model.theta,
            mu=args.model.mu,
            sigma=args.model.sigma,
            inter=args.model.inter,
            dt=args.model.dt,
            rtol=args.model.rtol,
            atol=args.model.atol,
            method=args.model.method,
            adaptive=args.model.adaptive,
        )
    else:
        raise NotImplementedError
    return model
