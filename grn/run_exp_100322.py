from pysched import scheduler as sch
import os
import numpy as np
from itertools import product


def generate_command():
    command = "/opt/jupyterhub/bin/python3 mutation_linobj.py -t {temperature} -x {sparsity} --name {name}"
    
    cmds = []
    for temp, sparsity, i in product([0.05, 0.1, 0.2,],
                                     [0.0, 0.1, 0.2, 0.5, 1.0],
                                     range(5)):
        name = f"exp_mut_100322_t{temp}_sp{sparsity}_id{i}"
        cmd = command.format(temperature=temp, sparsity=sparsity, name=name)
        cmds.append(cmd)
        
    return cmds


def setup_scheduler(cmds):
    sch.set_max_processes(4)
    for cmd in cmds:
        sch.register_command(cmd)

    sch.launch()


if __name__ == "__main__":
    cmds = generate_command()
    print(len(cmds))
    print(cmds[0])
    setup_scheduler(cmds)
