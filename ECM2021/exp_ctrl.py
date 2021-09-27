import sys
sys.path.append("..")
from packages import bashscheduler as bs
import os

def parse_name(submodel, addstr, sample):
    return f"expLI_{submodel}_{addstr}_n{sample}"

def generate_command():
    rootpy = "cd /home/nathan/Bureau/ENS/Thèse/dev/EmbryonicCortexModelling/;"
    basecmd = f"python3 ECM2021/run.py -n %s -e 90 -s 6 -m %s %s"
    redirect = " > output/logs/%s"

    all_cmds = []

    variates = [
        # ("basic", ""),
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
    bs.set_max_processes(8)
    for cmd in cmds:
        bs.register_command(cmd)

    bs.launch()
    bs.stop()  # TO REMOVE !!!


if __name__ == "__main__":
    cmds = generate_command()
    print(len(cmds))
    # print(cmds[0])
    setup_scheduler(cmds)
