import re

from torch import nn

from models.unet import UNet
from models.smp import SMPModel

ntype={'bn':nn.BatchNorm2d,
       'in':nn.InstanceNorm2d,
       'gn':nn.GroupNorm,
       'ln':nn.LayerNorm
      }

def get_hyp_params(module):
    args=[]
    kwargs={}
    repr_=module.__repr__()
    for string in re.search(r"(\()(.+)(\))",repr_)[2].split(","):
        if "=" in string:
            sep=string.strip(" ").split("=")
            kwargs[sep[0]]=eval(sep[1])
        else:
            args+=[eval(string)]
    return args,kwargs
def sub_dict(keys,dictionary):
    return {k:v for k,v in dictionary.items() if k in keys}

def batch_change_layers(model,model_config):
    target_type=model_config['norm']
    other_types=[v for k,v in ntype.items() if k!=target_type]
    def change_layer(sub):
        for n,c in sub.named_children():
            if type(c) in ntype.values():
                if type(c) in other_types:
                    hyp=get_hyp_params(c)
                    if target_type=="gn":
                        layer=ntype[target_type](num_groups=4,num_channels=hyp[0],**sub_dict(['eps','device','dtype'],hyp[1]))
                    elif target_type=="ln":
                        layer=ntype[target_type](normalized_shape=hyp[0],**sub_dict(['eps','device','dtype'],hyp[1]))
                    else:
                        layer=ntype[target_type](*hyp[0],**hyp[1])
                    _=sub.__setattr__(n,layer)
            _=change_layer(c)
    _=change_layer(model)
    
def build_model(config):
    model_type = config['model_type']
    model_config = config[model_type]
    num_cls = config['num_cls']
    aux = config['aux']
    print(model_type)
    if model_type == 'UNet':
        model = UNet(model_config, num_cls)
    elif model_type == 'smp':
        model = SMPModel(model_config, num_cls, aux)
    else:
        AssertionError('Wrong model type')
    if 'norm' in model_config:
        _=batch_change_layers(model,model_config)
    return model

if __name__ == '__main__':
    print('models/init')
    