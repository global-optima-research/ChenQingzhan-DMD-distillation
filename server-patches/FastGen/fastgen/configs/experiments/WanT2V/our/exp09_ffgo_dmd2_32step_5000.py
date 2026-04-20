# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from fastgen.configs.experiments.WanT2V.config_dmd2 import create_config as create_base_config
from fastgen.configs.net import Wan_1_3B_Config


def create_config():
    config = create_base_config()

    # Keep teacher/student architecture compatible until merged FFGO weights are ready.
    config.model.net = copy.deepcopy(Wan_1_3B_Config)
    config.model.teacher = copy.deepcopy(Wan_1_3B_Config)

    # Target a 32-step student rather than FastGen's default 4-step preset.
    config.model.student_sample_steps = 32
    config.model.sample_t_cfg.t_list = None

    # Stage run requested by the user.
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 50
    config.trainer.save_ckpt_iter = 500
    config.trainer.validation_iter = 500
    config.trainer.batch_size_global = 4
    config.dataloader_train.batch_size = 1

    config.log_config.group = "wan_ffgo_dmd2_feasibility"
    return config
