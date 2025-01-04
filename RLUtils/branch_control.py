import json
import os
import logging
class FallbackController(object):
    def __init__(self, load_path = None) -> None:
        self.fall_back_configs = {}
        if load_path is not None:
            self.load_fallback_json(load_path)
        


    def load_fallback_json(self, file, *args, **kwargs):
        if type(file) is str:
            with open(file, "r") as f:
                fall_back_config = json.load(f)
        else:
            fall_back_config = file
        self.fall_back_configs[fall_back_config['name']] = fall_back_config
    
    def get_fallback(self, config_name, item_name, branch, enable_default = False, enable_auto_fallback = False):
        assert len(branch) > 0
        if enable_default:
            fallback_level = self.fall_back_configs[config_name][item_name]["fallbacks"].get(branch, "default")
            plan = self.fall_back_configs[config_name][item_name][fallback_level]
            return plan
        elif enable_auto_fallback:
            if branch in self.fall_back_configs[config_name][item_name]["fallbacks"].keys():
                fallback_level = self.fall_back_configs[config_name][item_name]["fallbacks"].get(branch)
                plan = self.fall_back_configs[config_name][item_name][fallback_level]
                return plan
            else:
                fall_back_branch = branch[:-1]
                logging.warning(f"\033[33m RLUtils.FallbackController: Item \033[34m{config_name}.{item_name}\033[33m fallbacking from \033[34m{branch}\033[33m to \033[34m{fall_back_branch}\033[33m.\033[0m")
                return self.get_fallback(  config_name, item_name, fall_back_branch, enable_default, enable_auto_fallback)
            
        else:
            fallback_level = self.fall_back_configs[config_name][item_name]["fallbacks"].get(branch)
            plan = self.fall_back_configs[config_name][item_name][fallback_level]
            return plan