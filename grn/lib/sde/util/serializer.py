import json


def grn_from_json(json_str):
    json_dict = json.loads(json_str)
    if json_dict["object_name"] == "GRNOpt":
        from lib.sde.grn.grn import GRNOpt
        class_ = GRNOpt

    elif json_dict["object_name"] == "GRNMain":
        from lib.sde.grn.grn import GRNMain
        class_ = GRNMain

    elif json_dict["object_name"] == "GRNMain2":
        from lib.sde.grn.grn2 import GRNMain2
        class_ = GRNMain2

    elif json_dict["object_name"] == "GRNMain3":
        from lib.sde.grn.grn3 import GRNMain3
        class_ = GRNMain3

    elif json_dict["object_name"] == "GRNMain4":
        from lib.sde.grn.grn4 import GRNMain4
        class_ = GRNMain4
        grn = class_(nb_genes=json_dict["nb_genes"],
                     nb_membrane_gene=json_dict["nb_membrane_gene"],
                     nb_off_trees=json_dict["nb_off_trees"])

        grn.set_param(**json_dict)
        return grn

    else:
        raise ValueError(f"Class does not exist : {json_dict['object_name']}"
                         "You may have forgotten to add the class to the serializer "
                         "file if you are developing something")

    grn = class_(nb_genes=json_dict["nb_genes"],
                 nb_start_gene=json_dict["nb_start_gene"],
                 nb_off_trees=json_dict["nb_off_trees"])

    grn.set_param(**json_dict)
    return grn
