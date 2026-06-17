#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境变量配置加载器，用于读取 .env 文件中的配置。
"""
import os


def load_env_config() -> dict:
    """
    加载 .env 文件中的配置。

    Returns:
        dict: 包含所有环境变量配置的字典
    """
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    config = {}

    if os.path.exists(env_file):
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过注释和空行
                if not line or line.startswith('#') or '=' not in line:
                    continue

                # 分割键值对
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # 去除可能的引号
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                config[key] = value

    return config


def get_email_config() -> dict:
    """
    获取邮件配置。

    Returns:
        dict: 包含邮件发送者、密码、接收者的字典
    """
    config = load_env_config()
    return {
        'sender': config.get('EMAIL_SENDER', ''),
        'password': config.get('EMAIL_PASSWORD', ''),
        'receiver': config.get('EMAIL_RECEIVER', ''),
    }


def get_okx_config() -> dict:
    """
    获取 OKX 配置。

    Returns:
        dict: 包含 API key, secret, passphrase 的字典
    """
    config = load_env_config()
    return {
        'api_key': config.get('OKX_API_KEY', ''),
        'secret_key': config.get('OKX_SECRET_KEY', ''),
        'passphrase': config.get('OKX_PASSPHASE', ''),
    }


# 提供全局配置变量
EMAIL_CONFIG = get_email_config()
OKX_CONFIG = get_okx_config()
