from pysched import scheduler as sch
import os
import numpy as np


def generate_command():
    all_commands = """
    /opt/jupyterhub/bin/python3 script_multi_param.py -n 5 -t 0.5 -g 30 -m 3 --norm 1.5
    /opt/jupyterhub/bin/python3 script_multi_param.py -n 7 -t 0.5 -g 30 -m 3 --norm 1.5
    /opt/jupyterhub/bin/python3 script_multi_param.py -n 9 -t 0.5 -g 30 -m 3 --norm 1.5
    /opt/jupyterhub/bin/python3 script_multi_param.py -n 7 -t 0.25 -g 30 -m 3 --norm 1.5
    /opt/jupyterhub/bin/python3 script_multi_param.py -n 7 -t 1.0 -g 30 -m 3 --norm 1.5
    /opt/jupyterhub/bin/python3 script_multi_param.py -n 7 -t 0.5 -g 30 -m 1 --norm 1.5
    /opt/jupyterhub/bin/python3 script_multi_param.py -n 7 -t 0.5 -g 30 -m 5 --norm 1.5
    /opt/jupyterhub/bin/python3 script_multi_param.py -n 7 -t 0.5 -g 30 -m 3 --norm 1.0
    /opt/jupyterhub/bin/python3 script_multi_param.py -n 7 -t 0.5 -g 30 -m 3 --norm 2.0
    """
    
    raw_cmd = all_commands.strip().split('\n')
    cmds = []
    for i in range(5):
        cmds += list(map(lambda x: x + f" -i {i}", raw_cmd))
        
    return cmds


def setup_scheduler(cmds):
    sch.set_max_processes(8)
    for cmd in cmds:
        sch.register_command(cmd)

    sch.launch()


if __name__ == "__main__":
    cmds = generate_command()
    print(len(cmds))
    print(cmds[0])
    setup_scheduler(cmds)
