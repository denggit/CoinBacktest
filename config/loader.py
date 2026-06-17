#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import yaml


# 为了类型提示


# 如果系统还有地方强依赖旧的 settings.yaml (如 API Key)，可以保留这部分：
def load_global_settings():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    settings_path = os.path.join(current_dir, 'settings.yaml')
    if os.path.exists(settings_path):
        with open(settings_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    return {}


GLOBAL_SETTINGS = load_global_settings()
