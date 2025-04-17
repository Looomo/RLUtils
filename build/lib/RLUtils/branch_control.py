import json
import os
import logging
class FallbackController(object):
    def __init__(self, load_path = None) -> None:
        self.fall_back_configs = {}
        if load_path is not None:
            self.load_fallback_json(load_path)
        
        self.set_global_auto_fallback(False)
        self.set_global_enable_default(False)

    def set_global_auto_fallback(self, global_auto_fallback):
        self.global_auto_fallback = global_auto_fallback
        return
    
    def set_global_enable_default(self, enable_default):
        self.enable_default = enable_default
        return

    def load_fallback_json(self, file, *args, **kwargs):
        if type(file) is str:
            with open(file, "r") as f:
                fall_back_config = json.load(f)
        else:
            fall_back_config = file
        self.fall_back_configs[fall_back_config['name']] = fall_back_config
    
    def get_fallback(self, config_name, item_name, branch, enable_default = False, enable_auto_fallback = False):
        if self.global_auto_fallback:
            logging.warning(f"\033[33m RLUtils.FallbackController: enable_auto_fallback set to \033[34m True \033[33m, as global_auto_fallback flag is \033[34m True. \033[33m\033[0m")
            enable_auto_fallback = enable_auto_fallback or self.global_auto_fallback
        # assert len(branch) > 0
        if len(branch) == 0: 
            return self.get_default(config_name, item_name, branch) if (enable_default or self.enable_default) else None

        if branch in self.fall_back_configs[config_name][item_name]["fallbacks"].keys():
            fallback_level = self.fall_back_configs[config_name][item_name]["fallbacks"].get(branch)
            plan = self.fall_back_configs[config_name][item_name][fallback_level]
            return plan
        
        # if branch is not registered 

        if enable_auto_fallback:
            fall_back_branch = branch[:-1]
            logging.warning(f"\033[33m RLUtils.FallbackController: Item \033[34m{config_name}.{item_name}\033[33m fallbacking from \033[34m{branch}\033[33m to \033[34m{fall_back_branch}\033[33m.\033[0m")
            return self.get_fallback(  config_name, item_name, fall_back_branch, enable_default, enable_auto_fallback)
        
        if enable_default: return self.get_default(config_name, item_name, branch)
            
        
    def get_default(self, config_name, item_name, branch):
        fallback_level = self.fall_back_configs[config_name][item_name]["fallbacks"].get(branch, "default")
        if fallback_level == "default":
            logging.warning(f"\033[1;31;43m Default Warning: Item \033[34m{config_name}.{item_name} fallbacks to default, please check ↓")
        logging.warning(f"\033[33m RLUtils.FallbackController: Item \033[34m{config_name}.{item_name}\033[33m fallbacking from \033[34m{branch}\033[33m to \033[34m default \033[33m.\033[0m")
        plan = self.fall_back_configs[config_name][item_name][fallback_level]
        return plan
