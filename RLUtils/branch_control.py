import json
import os
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
    
    def get_fallback(self, config_name, item_name, branch):
        
        fallback_level = self.fall_back_configs[config_name][item_name]["fallbacks"][branch]
        plan = self.fall_back_configs[config_name][item_name][fallback_level]
        return plan