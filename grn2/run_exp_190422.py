from pysched import scheduler as sch
import os
import numpy as np
from itertools import product


def generate_command():
    command = "/opt/jupyterhub/bin/python3 opti_metaparam1.py -n {name} -v {value} -i {id} > out.txt"
    
    cmds = []
    runs = [
        ("ctrl", 0.),
        ("temp_param", 0.05),
        ("temp_param", 0.2),
        ("sparse_param", 0.1),
        ("sparse_param", 0.3),
        ("temp_gene", 0.05),
        ("temp_gene", 0.2),
        ("sparse_gene", 0.1),
        ("sparse_gene", 0.3),
        ("decay_temp_param", 0.),
        ("decay_temp_gene", 0.),
        ("number_gene", 0.),
    ]
    for (name, value), i in product(runs,
                                     range(5)):
        cmd = command.format(name=name, value=value, id=i)
        cmds.append(cmd)
        
    return cmds


def setup_scheduler(cmds):
    sch.set_max_processes(6)
    for cmd in cmds:
        sch.register_command(cmd)

    sch.launch()


if __name__ == "__main__":
    cmds = generate_command()
    print(len(cmds))
    print(cmds[0])
    setup_scheduler(cmds)
