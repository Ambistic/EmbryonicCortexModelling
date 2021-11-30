import sys
sys.path.append("/home/nathan/other/thesis_nathan/pysched/")
from pysched import scheduler as sch
import os

def parse_name(submodel, addstr, sample):
    return f"expLI_{submodel}_{addstr}_n{sample}"

def generate_command():
    rootpy = f"cd {os.getcwd()};"
    basecmd = f"python3 run.py -n %s -e 90 -s 6 -m %s %s"
    redirect = " > ../output/logs/%s 2>&1"

    all_cmds = []

    variates = [
        ("basic", ""),
        ("tristateLI", ""),
        ("tristate1", ""),
        ("bistate1", ""),
    ]

    for v in variates:
        name = parse_name(v[0], v[1], 1)
        cmd = rootpy + basecmd % (name, v[0], v[1]) + redirect % (name + ".txt")
        all_cmds.append(cmd)

    return all_cmds

def keep_name(name):
    # check if csv is there
    if os.path.exists("output/results/stats_" + name + ".csv"):
        return False
    return True

def setup_scheduler(cmds):
    sch.set_max_processes(8)
    for cmd in cmds:
        sch.register_command(cmd)

    sch.launch()
    # bs.stop()  # TO REMOVE !!!


if __name__ == "__main__":
    cmds = generate_command()
    print(len(cmds))
    # print(cmds[0])
    setup_scheduler(cmds)
