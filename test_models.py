import yaml
import numpy as np
import mindspore as ms
import mindocr
from mindocr.models.backbones import build_backbone
from mindocr.models import build_model
from mindspore import load_checkpoint, load_param_into_net


with open('./config_sample1.yaml') as fp:
    model_cfg1 = yaml.safe_load(fp)['Architecture']

with open('./config_sample2.yaml') as fp:
    model_cfg2 = yaml.safe_load(fp)['Architecture']


def test_registry():
    #mindocr.models.backbones.list_backbones
    print('Backbone list', mindocr.list_backbones())
    print('Backbone class list', mindocr.list_backbone_classes())
    print('Model list: ', mindocr.list_models())

#@pytest.mark.parametrize('pretrained', [False, True])
def test_backbone(pretrained=False):
    #cfg = model_cfg1  

    # configuration methods 
    cfg_from_predef = dict(name='det_resnet50', pretrained=False)

    cfg_from_class  = dict(name='DetResNet', layers=[3,4,6,3], out_indices=[1,2,3])
    
    # forward
    ms.set_context(mode=ms.PYNATIVE_MODE)
    bs = 8
    image_size = 224
    dummy_input = ms.Tensor(np.random.rand(bs, 3, image_size, image_size), dtype=ms.float32)

    for cfg in [cfg_from_predef, cfg_from_class]:
        backbone = build_backbone(cfg)
        #import mindcv
        #backbone = mindcv.create_model('resnet50')
        
        ftrs = backbone(dummy_input)
        print('output shape: ', [ftr.shape for ftr in ftrs])

        assert ftrs[-1].shape[0]==bs, 'output shape not match'

def test_model():
    # 1. by model name 
    model = build_model('dbnet_r50', pretrained=False)


    # 2. by arch config
    model_config = {
            "backbone": {
                'name': 'det_resnet50',
                'pretrained': False 
                },
            "neck": {
                "name": 'FPN',
                "out_channels": 256,
                },
            "head": {
                "name": 'ConvHead',
                "out_channels": 2,
                "k": 50
                }
            
            }

    model = build_model(model_config)
    
    ckpt_path = None   
    if ckpt_path is not None:
        param_dict = load_checkpoint(os.path.join(path, os.path.basename(default_cfg['url'])))
        load_param_into_net(model, param_dict)



if __name__ == '__main__':    
    test_registry()
    #test_backbone()
    test_model()
