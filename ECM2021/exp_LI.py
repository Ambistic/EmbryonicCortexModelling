from packages import bashscheduler as bs
import os

def parse_name(submodel, key, value, sample):
    return f"expLI_{submodel}_{key}_{value}_n{sample}"

def generate_command():
    rootpy = "/home/nathan/Bureau/ENS/Thèse/dev/EmbryonicCortexModelling/"
    basecmd = f"python3 AB-GPN-LI.py -n %s -e 94 %s -s 8 " \
            "--bias_1 %f --bias_2 %f --bias_3 %f --bias_4 %f " \
            "--bias_5 %f -v %f -o %f -p %d --check " \
            " > output/%s"

    keys = ["name", "gpasip", "b1", "b2", "b3", "b4", "b5",
            "startval", "smooth", "sample", "name"]
    dict_args = {
            "name": None,
            "gpasip": "-g",
            "b1": 0.2,
            "b2": 0.15,
            "b3": -0.1,
            "b4": -0.1,
            "b5": -0.1,
            "startval": 0.35,
            "smooth": 0.,
    }

    all_cmds = []

    variates = [
            ("ctrl", ""),
            ("gpasip", ""),

            ("smooth", 0.05),
            ("smooth", 0.1),
            ("smooth", 0.15),
            ("smooth", 0.2),

            ("startval", 0.25),
            ("startval", 0.3),
            ("startval", 0.40),
            ("startval", 0.45),

            ("b1", 0.1),
            ("b1", 0.15),
            ("b1", 0.25),
            ("b1", 0.3),

            ("b2", 0.05),
            ("b2", 0.1),
            ("b2", 0.2),
            ("b2", 0.25),

            ("b3", -0.2),
            ("b3", -0.15),
            ("b3", -0.05),
            ("b3", 0.),

            ("b4", -0.2),
            ("b4", -0.15),
            ("b4", -0.05),
            ("b4", 0.),

            ("b5", -0.2),
            ("b5", -0.15),
            ("b5", -0.05),
            ("b5", 0.),
    ]

    for v in variates:
        for sample in range(1, 5 + 1):
            d = dict_args.copy()
            name = parse_name(v[0], v[1], sample)
            d["name"] = name
            d[v[0]] = v[1]
            d["sample"] = sample
            tup = tuple(d[k] for k in keys)
            cmd = basecmd % tup
            if keep_name(name):
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


if __name__ == "__main__":
    cmds = generate_command()
    print(len(cmds))
    # print(cmds[0])
    setup_scheduler(cmds)
