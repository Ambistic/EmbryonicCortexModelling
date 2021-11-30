import sys
sys.path.append("/home/nathan/other/thesis_nathan/pysched/")
from pysched import scheduler as sch
import os
import numpy as np
from lib.stringmodel import StringModel

def parse_name(submodel, addstr, sample, end, size):
    return f"expcomp_{submodel}_e{end}_s{size}_{addstr.replace(' ', '_')}_n{sample}"

def generate_command():
    nb_sample = 5
    rootpy = f"cd {os.getcwd()};"
    # basecmd = f"python3 run.py -n %s -e 90 -s 6 -m %s %s"
    basecmd = "python3 run.py -n {name} -e {end} -s {size} -t {start} -m {model} -i {sample} {params}".format
    redirect = " > ../output/logs/%s 2>&1"
    
    default = dict(model="triambimutant", end=90, size=8, sample=1, start=49, params="")
    
    sm = StringModel(
        "expliis_m{model}_e{end}_s{size}_p{params}_n{sample}_t{start}",
        default=default,
        forbidden=" ",
    )

    all_cmds = []

    variates = [
        dict(params=f"-p smooth {smooth}")
        for smooth in np.arange(0, 1.1, 0.2)
    ]

    for v in variates:
        for i in range(nb_sample):
            v["sample"] = i
            v["name"] = sm.fill(**v)
            param = default.copy()
            param.update(v)
            cmd = rootpy + basecmd(**param) + redirect % (param["name"] + ".txt")
            all_cmds.append(cmd)

    return all_cmds

def keep_name(name):
    # check if csv is there
    if os.path.exists("output/results/" + name + ".stats.csv"):
        return False
    return True

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
