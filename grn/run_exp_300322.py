from pysched import scheduler as sch
import os
import numpy as np
from itertools import product


def generate_command():
    command = "/opt/jupyterhub/bin/python3 script_exp_li_total3_300322.py -t {type} --name {name}"
    
    cmds = []
    for type, i in product(["normal", "fixednoise", "nonoise", "noasym", "nointercell",
                           "nothing", 'onlyintercell', "onlyasym", "onlynoise"],
                                     range(10)):
        name = f"exp_li_total_300322_t{type}_id{i}"
        cmd = command.format(type=type, name=name)
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
