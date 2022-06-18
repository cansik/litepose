from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import json

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.multiprocessing
import torch.onnx
from torch.autograd import Variable

import lib.models.pose_mobilenet
from lib.config import cfg
from lib.config import check_config
from lib.config import update_config
from lib.fp16_utils.fp16util import network_to_half
from lib.utils.utils import create_logger
from lib.utils.utils import get_model_summary
from arch_manager import ArchManager

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # fixed config for supernet
    parser.add_argument('--superconfig',
                        default=None,
                        type=str,
                        help='fixed arch for supernet training')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    # change the resolution according to config
    fixed_arch = None
    if args.superconfig is not None:
        with open(args.superconfig, 'r') as f:
            fixed_arch = json.load(f)
        cfg.defrost()
        reso = fixed_arch['img_size']
        cfg.DATASET.INPUT_SIZE = reso
        cfg.DATASET.OUTPUT_SIZE = [reso // 4, reso // 2]
        cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.MODEL.NAME == 'pose_mobilenet' or cfg.MODEL.NAME == 'pose_simplenet':
        arch_manager = ArchManager(cfg)
        cfg_arch = arch_manager.fixed_sample()
        if fixed_arch is not None:
            cfg_arch = fixed_arch
        model = lib.models.pose_mobilenet.get_pose_net(cfg, is_train=True, cfg_arch=cfg_arch)
        # model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        #     cfg, is_train=True, cfg_arch=cfg_arch
        # )
    else:
        model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
            cfg, is_train=True
        )

    # set super config
    if cfg.MODEL.NAME == 'pose_supermobilenet':
        model.arch_manager.is_search = True
        if args.superconfig is not None:
            with open(args.superconfig, 'r') as f:
                model.arch_manager.search_arch = json.load(f)
        else:
            model.arch_manager.search_arch = model.arch_manager.fixed_sample()

    dump_input = torch.rand(
        (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
    )
    logger.info(get_model_summary(cfg.DATASET.INPUT_SIZE, model, dump_input))

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device('cpu')), strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth.tar'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    print(model)

    print("exporting...")
    dummy_input = Variable(torch.randn(1, 3, 256, 256))
    torch.onnx.export(model, dummy_input, f"LitePose-Auto-XS.onnx")
    print("done")


if __name__ == '__main__':
    main()
