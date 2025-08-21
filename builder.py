import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder,TimmCNNEncoder_lxq
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH
        
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder(model_name='resnet50',
            kwargs={
                'features_only': True,
                'out_indices': (3,),
                'pretrained': True,
                'num_classes': 0
            },
            pool=True)
    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == 'conch_v1_5':
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("Please install huggingface transformers (e.g. 'pip install transformers') to use CONCH v1.5")
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        model, _ = titan.return_conch()
        assert target_img_size == 448, 'TITAN is used with 448x448 CONCH v1.5 features'
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms

def get_encoder_lxq(model_name: str, target_img_size: int = 224, os: int = 32, 
                return_stem: bool = False, pool: bool = False):
    """
    返回
    - model: 可输出多尺度特征的编码器（dict: {'c2','c3','c4','c5', 可选 'c1'}）
    - img_transforms: 与该模型预训练配置匹配的eval变换
    参数
    - model_name: 'resnet50_trunc' 或 'uni_v1'（保持你的分支）
    - target_img_size: patch输入大小（仅用于预处理变换）
    - os: 输出步幅（32/16/8），越小越细致但耗显存
    - return_stem: 是否返回 /4 的 stem 特征作为额外 skip（c1）
    - pool: 若为 True，会额外返回 'token'（对最高层特征做 GAP）
    """
    print('loading model checkpoint')

    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder_lxq(
            model_name='resnet50.tv_in1k',
            os=os,
            return_stem=return_stem,
            pool=pool
        )
        # 用模型自身的预训练配置生成 transforms
        cfg = resolve_data_config(model.backbone.pretrained_cfg)
        cfg['input_size'] = target_img_size
        img_transforms = create_transform(
            **cfg, is_training=False)
        return model, img_transforms

    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model(
            "vit_large_patch16_224",
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True
        )
        cfg = resolve_data_config(model.pretrained_cfg)
        img_transforms = create_transform(
            **cfg, is_training=False, input_size=target_img_size
        )
        return model, img_transforms