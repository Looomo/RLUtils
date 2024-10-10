from typing import Any
import torch
import os
# from params_proto.neo_proto import ParamsProto, PrefixProto, Proto
import json
import copy
import pickle as pkl
class SharedConfigs:
    save_path_template = {
    }
    _dict = {}
    logbase = "logs"
    # save_path_template = 'f"{Configs.logbase}"'
    m_tag: None
    def set_from_dict(data_dict):
        SharedConfigs._dict = data_dict
        for key, value in data_dict.items():
            setattr(SharedConfigs, key, value)

    def __call__():
        return SharedConfigs._dict
    
    def eval_path(template, args):
        template = "f'" + template + "'"
        args = SharedConfigs.args
        # evaled = [ eval( para ) for para in template ]
        save_path =  eval(template) # os.path.join(*evaled)
        os.makedirs( save_path, exist_ok=True )
        # Configs.save_path = save_path
        SharedConfigs.add_extra("save_path", save_path)

    def add_extra(key, value):
        SharedConfigs._dict[key] = value
        setattr(SharedConfigs, key, value)

    def savecfg():
        params = copy.deepcopy(SharedConfigs._dict)
        for k in params.keys():
            params[k] = str(params[k])
        filepath = os.path.join(SharedConfigs.save_path, f'config.json')
        with open(filepath , 'w') as fp:
            json.dump(params, fp, indent=4)


        with open(os.path.join(SharedConfigs.save_path, f'args.pkl'), "wb") as f:
            pkl.dump(SharedConfigs.args, f)
        SharedConfigs.logger.info(f'Configs is saved to bucket: {filepath}')