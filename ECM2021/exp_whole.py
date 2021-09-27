import sys
from pathlib import Path as P
sys.path.append("..")
from packages import bashscheduler as bs
import os

def parse_name(submodel, addstr, sample):
    addstr = addstr.replace(" ", "_").replace("--", "_").replace("__", "_")
    return f"expLI_{submodel}_{addstr}_n{sample}"

def metavariate_to_variate(meta):
    ls = list()
    for i, v in enumerate(meta[2]):
        ls.append((meta[0], meta[1] + "_" + str(i + 1), v, meta[3]))
    return ls

def variate_to_instance(variate):
    ls = list()
    for c in [-2, -1, 1, 2]:
        ls.append((variate[0], f"--{variate[1]} %.3f" % (variate[2] + c * variate[3])))
    return ls

def instance_to_sample(instance):
    ls = list()
    for i in range(5):
        ls.append((instance[0], instance[1], i + 1))
    return ls

def generate_command():
    rootpy = "cd /home/nathan/Bureau/ENS/Thèse/dev/EmbryonicCortexModelling/;"
    basecmd = f"python3 ECM2021/run.py -n %s -e 90 -s 6 -m %s -p %d %s"
    redirect = " > output/logs/%s"

    all_cmds = []
    
    metavariates = [  # model, name, defval, step
        ("tristateLI", "diff_values_IP", [0.23, 0.23, 0.23, 0.23], 0.025),
        ("tristateLI", "bias_ratio", [0.2, 0.15, -0.1, -0.1], 0.05),
        ("tristateLI", "tc_coeff_RG", [1., 1., 1., 1.], 0.1),
        ("tristateLI", "tc_coeff_IP", [1., 1., 1., 1.], 0.1),
        
        ("tristate1", "diff_values_IP", [0.23, 0.23, 0.23, 0.23], 0.025),
        ("tristate1", "diff_values_RG_IP", [0.63, 0.53, 0.43, 0.38], 0.025),
        ("tristate1", "tc_coeff_RG", [1., 1., 1., 1.], 0.1),
        ("tristate1", "tc_coeff_IP", [1., 1., 1., 1.], 0.1),
        
        ("bistate1", "diff_values_IP", [0.23, 0.23, 0.23, 0.23], 0.025),
        ("bistate1", "diff_values_RG", [0.63, 0.53, 0.43, 0.38], 0.025),
        ("bistate1", "tc_coeff_RG", [1., 1., 1., 1.], 0.1),
        ("bistate1", "tc_coeff_IP", [1., 1., 1., 1.], 0.1),
        
        ("basic", "diff_values", [0.73, 0.63, 0.53, 0.43], 0.025),
        ("basic", "tc_coeff", [1., 1., 1., 1.], 0.1),
    ]
    
    variates = [ # model, name, defval, step
        ("tristate1", "diff_values_RG_GP_4", 0.8, 0.025),
        ("tristate1", "diff_values_RG_GP_5", 0.6, 0.025),
        ("tristateLI", "startval", 0.35, 0.05),
    ]
    
    instances = [
        ("basic", ""),
        ("bistate1", ""),
        ("tristate1", ""),
        ("tristateLI", ""),
        ("tristateLI", "-g"),
        ("tristateLI", "--smooth 0.10"),
        ("tristateLI", "--smooth 0.20"),
    ]

    samples = [ # model, add_param, sample
        
    ]
    
    for m in metavariates:
        variates += metavariate_to_variate(m)
        
    for v in variates:
        instances += variate_to_instance(v)
        
    for ins in instances:
        samples += instance_to_sample(ins)

    for v in samples:
        name = parse_name(*v)
        if not keep_name(name):
            continue
        cmd = rootpy + basecmd % (name, v[0], v[2], v[1]) + redirect % (name + ".txt")
        all_cmds.append(cmd)

    return all_cmds

def keep_name(name):
    # check if csv is there
    ROOT = P(os.path.realpath(__file__)).parent.parent
    if os.path.exists(ROOT / f"output/results/stats_{name}.csv"):
        return False
    return True

def setup_scheduler(cmds):
    bs.set_max_processes(8)
    for cmd in cmds:
        bs.register_command(cmd)

    bs.launch()
    # bs.stop()  # TO REMOVE !!!


if __name__ == "__main__":
    cmds = generate_command()
    print(f"Let's go for {len(cmds)} runs !")
    print(cmds[-5:])
         
    setup_scheduler(cmds)
