"""
Consolidate the model architecture definition to allow loading elsewhere.
Author: Alicia Matsumoto
"""
import argparse
import torch
import torch.nn as nn
from models_istt import build_model
from main_style_transfer import get_args_parser


def get_model_and_criterion():
    parser = argparse.ArgumentParser(
        'DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    model, criterion, postprocessors = build_model(args)
    return model, criterion


if __name__ == "__main__":
    model, criterion = get_model_and_criterion()
    checkpoint = {'model': model,
                  'criterion': criterion}
    torch.save(checkpoint, 'arch_to_load.pth')
