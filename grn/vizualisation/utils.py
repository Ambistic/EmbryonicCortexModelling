from collections import defaultdict
import os
from lib.stringmodel import StringModel

default = dict(model="triambimutant", end=90, size=8, sample=1, start=49, params="")
sm = StringModel(
    "exp_m{model}_e{end}_s{size}_p{params}_n{sample}_t{start}",
    default=default,
    sep="_",
    forbidden=" ",
)


def dict_to_listtuple(x):
    return [(k, v) for k, v in x.items()]


def listtuple_to_dict(x):
    return {k: v for k, v in x}


def get_param_name(key):
    return key[2]


def get_name_for_file(key):
    base = "_".join([x[1] for x in key[0]]) + "_"
    params = "".join(["".join(x) for x in key[1]])
    end = "$_" + key[2]
    return base + params + end


def get_splitted_params(params):
    ls = params.split("-p")[1:]
    return tuple(map(lambda x: tuple(x.split("%")[1:]), ls))


def extract_param_from_name(name):
    name = ".".join(name.split(".")[:-2])
    dc = sm.extract(name)  # TODO split the -p
    dc["params"] = get_splitted_params(dc["params"])
    dc["name"] = name
    return dc


def read_dir(root):
    all_params = list()
    for path, subdirs, files in os.walk(root):
        for file in files:
            if ".stats.csv" in file:
                all_params.append(extract_param_from_name(file))

    return all_params


def gather_groups(params):
    groups = defaultdict(list)
    for param in params:
        key = tuple((k, v) for k, v in param.items() if k not in ['sample', 'name'])
        groups[key].append(param)
    return groups


def gather_metagroups(groups, keep_ones=True):
    metagroups = defaultdict(dict)
    exp = {key_group: value_group for key_group, value_group in groups.items()}

    for key_group, value_group in exp.items():
        ful_key_dict = listtuple_to_dict(key_group)
        key_static = tuple(dict_to_listtuple({k: v for k, v in ful_key_dict.items()
                                              if k not in ["params", "name"]}))

        # because sometimes a third empty component is found
        ful_key_dict["params"] = [x[:2] for x in ful_key_dict["params"]]

        dict_params = listtuple_to_dict(ful_key_dict["params"])
        for non_param, non_param_value in ful_key_dict["params"]:
            unselected = tuple(dict_to_listtuple({k: v for k, v in dict_params.items() if k != non_param}))
            key_meta = (key_static, unselected, non_param)
            metagroups[key_meta][non_param_value] = value_group

        if keep_ones:
            unselected = tuple(dict_to_listtuple({k: v for k, v in dict_params.items()}))
            key_meta = (key_static, unselected, "INDIV")
            metagroups[key_meta]["Control"] = value_group

    if not keep_ones:
        metagroups = {k: v for k, v in metagroups.items() if len(v) > 1}

    return metagroups
