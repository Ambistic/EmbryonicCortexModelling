import json


def grn_from_json(json_str):
    json_dict = json.loads(json_str)
    if json_dict["object_name"] == "GRNMain":
        from lib.sde.grn.grn import GRNMain
        class_ = GRNMain

    else:
        raise ValueError(f"Class does not exist : {json_dict['object_name']}"
                         "You may have forgotten to add the class to the serializer "
                         "file if you are developing something")

    grn = class_(nb_genes=json_dict["nb_genes"],
                 nb_membrane_gene=json_dict["nb_membrane_gene"],
                 nb_regulators=json_dict["nb_regulators"])

    grn.set_param(**json_dict)
    return grn
