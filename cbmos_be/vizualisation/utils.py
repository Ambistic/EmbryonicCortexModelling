from collections import defaultdict
import os
from .defaults import get_default_val

def dict_to_listtuple(x):
    return [(k, v) for k, v in x.items()]

def listtuple_to_dict(x):
    return {k: v for k, v in x}

def get_param_name(key):
    return [x[1] for x in key if x[0] == "param"][0].strip("_")

def get_name_for_file(key):
    return "_".join([x[1].strip("_") for x in key])

def extract_param_from_name(name):
    name = "_".join(name.split("_")[1:])
    name = ".".join(name.split(".")[:-1])
    shorten = name
    # shorten = name.replace("__", "_")
    
    exp, model, *param, value, n = shorten.split("_")
    param = "_".join(param)
    return dict(exp=exp, model=model, param=param, value=value, n=n, name=name)

def read_dir(root):
    all_params = list()
    for path, subdirs, files in os.walk(root):
        for file in files:
            if file.startswith("stats_"):
                all_params.append(extract_param_from_name(file))
                
    return all_params

def gather_groups(params):
    groups = defaultdict(list)
    for param in params:
        key = tuple((k, v) for k, v in param.items() if k in ["exp", "model", "param", "value"])
        groups[key].append(param)
    return groups

def gather_metagroups(groups):
    metagroups = defaultdict(dict)
    controls = {listtuple_to_dict(key_group)["model"]: value_group for key_group, value_group in groups.items()
               if listtuple_to_dict(key_group)["param"] == ""}
    exp = {key_group: value_group for key_group, value_group in groups.items()
               if listtuple_to_dict(key_group)["param"] != ""}
    
    for key_group, value_group in exp.items():            
        key_meta = tuple((k, v) for k, v in key_group if k in ["exp", "model", "param"])
        value = {k: v for k, v in key_group}["value"]
        metagroups[key_meta][value] = value_group
        
    for key_meta in metagroups:
        model = listtuple_to_dict(key_meta)["model"]
        param = listtuple_to_dict(key_meta)["param"]
        defval = str(get_default_val(param))
        
        metagroups[key_meta][defval] = controls[model]
    
    return metagroups