# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastgen.configs.experiments.WanT2V.config_mf_video import create_config as create_base_config


def create_config():
    config = create_base_config()

    # Align the run budget with the CD/DMD2 comparison runs.
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 50
    config.trainer.save_ckpt_iter = 500
    config.trainer.validation_iter = 500
    config.trainer.batch_size_global = 2
    config.dataloader_train.batch_size = 1

    config.log_config.group = "wan_mf_stage"
    return config
