#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import os
import sys
import platform
import yaml
import time
import datetime
from tqdm import tqdm
import cv2
import numpy as np
from argparse import ArgumentParser, RawDescriptionHelpFormatter

class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")
        self.add_argument(
            '-p',
            '--profiler_options',
            type=str,
            default=None,
            help='The option of profiler, which should be in format ' \
                 '\"key1=value1;key2=value2;key3=value3\".'
        )

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return config


def merge_config(config, opts):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in opts.items():
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in config
            ), "the sub_keys can only be one of global_config: {}, but get: " \
               "{}, please check your running command".format(
                config.keys(), sub_keys[0])
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    return config


def check_device(use_gpu, use_xpu=False, use_npu=False, use_mlu=False):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    err = "Config {} cannot be set as true while your paddle " \
          "is not compiled with {} ! \nPlease try: \n" \
          "\t1. Install paddlepaddle to run model on {} \n" \
          "\t2. Set {} as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and use_xpu:
            print("use_xpu and use_gpu can not both be ture.")
        if use_gpu and not paddle.is_compiled_with_cuda():
            print(err.format("use_gpu", "cuda", "gpu", "use_gpu"))
            sys.exit(1)
        if use_xpu and not paddle.device.is_compiled_with_xpu():
            print(err.format("use_xpu", "xpu", "xpu", "use_xpu"))
            sys.exit(1)
        if use_npu and not paddle.device.is_compiled_with_npu():
            print(err.format("use_npu", "npu", "npu", "use_npu"))
            sys.exit(1)
        if use_mlu and not paddle.device.is_compiled_with_mlu():
            print(err.format("use_mlu", "mlu", "mlu", "use_mlu"))
            sys.exit(1)
    except Exception as e:
        pass


def to_float32(preds):
    if isinstance(preds, dict):
        for k in preds:
            if isinstance(preds[k], dict) or isinstance(preds[k], list):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], paddle.Tensor):
                preds[k] = preds[k].astype(paddle.float32)
    elif isinstance(preds, list):
        for k in range(len(preds)):
            if isinstance(preds[k], dict):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], list):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], paddle.Tensor):
                preds[k] = preds[k].astype(paddle.float32)
    elif isinstance(preds, paddle.Tensor):
        preds = preds.astype(paddle.float32)
    return preds

