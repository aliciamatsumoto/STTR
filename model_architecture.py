"""
Consolidate the model architecture definition to allow loading elsewhere.
Author: Alicia Matsumoto
"""
import argparse
import torch
import torch.nn as nn
from models_istt import build_model
from demo_sttr_image_ACCV import get_args_parser


def get_model_and_criterion():
    parser = argparse.ArgumentParser(
        'DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args([
                          "--batch_size","1",
                          "--style_loss_coef","10",
                          "--fold_k","5",
                          "--lr_backbone","1e-5",
                          "--lr","1e-5",
                          "--enc_layers","6",
                          "--dec_layers","6",
                          "--model_type","nofold",
                          "--enorm","--dnorm","--tnorm",
                          "--cbackbone_layer","2","--sbackbone_layer","4",
                          
                          "--dataset_file","demo",
                          "--resume","checkpoint_model/checkpoint0005.pth"  ,
                          "--img_size","512",
                          "--in_content_folder","inputs/content",
                          "--style_folder","inputs/style",
                          "--output_dir","outputs",
                             ])
    model, criterion, postprocessors = build_model(args)
    return model, criterion


if __name__ == "__main__":
    model, criterion = get_model_and_criterion()
    checkpoint = {'model': model,
                  'criterion': criterion}
    torch.save(checkpoint, 'arch_to_load.pth')
