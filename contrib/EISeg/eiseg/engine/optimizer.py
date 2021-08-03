import paddle
import math
from paddleseg.utils import logger
import paddle.nn as nn

def get_optimizer_lr(model):
    backbone_params = nn.ParameterList()
    other_params = nn.ParameterList()
    for name, param in model.named_parameters():
        if 'ocr' in name:
            other_params.append(param)
            continue
        elif 'feature_extractor' not in name:
            other_params.append(param)
            continue
        else:
            backbone_params.append(param)

    return backbone_params, other_params
   