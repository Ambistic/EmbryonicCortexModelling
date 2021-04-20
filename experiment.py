import numpy as np
from types import SimpleNamespace
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random


def randomize():
    np.random.random()
    random.random()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def stop(ls):
    for x in ls:
        try:
            x.get(0)
        except:
            pass

def _run_one(args=(), kwargs=dict(), seed=0):
    if seed is not None:
        set_seed(seed)
    from model import Brain
    sample = Brain(*args, **kwargs)
    sample.run()
    x = SimpleNamespace()
    x.stats = sample.stats
    return x

def _safe_run_one(args=(), kwargs=dict(), seed=None):
    if seed is not None:
        set_seed(seed)
    from model import Brain
    trial = 0
    while trial < 5:
        trial += 1
        try:
            sample = Brain(*args, **kwargs)
            sample.run()
        except Exception as e:
            print("Missed trial", trial, e)
        else:
            x = SimpleNamespace()
            x.stats = sample.stats
            return x
    raise RuntimeError("Could not get the Brain to work after 5 trials")


class Experiment:
    def __init__(
        self, *args, number=10, cls_=None, safe_mode=True, max_number=100, **kwargs
    ):
        self.args = args
        self.kwargs = kwargs
        self.cls_ = cls_
        self.number = number
        self.max_number = max_number
        self.safe_mode = safe_mode

    def run_parallel(self, condition=None):
        self.samples = list()
        func = _safe_run_one if self.safe_mode else _run_one
        args, kwargs = self.args, self.kwargs
        count = 0
        max_count = self.number if condition is None else self.max_number

        with Pool(8) as p:
            ls_res = []
            for i in range(max_count):
                randomize()
                seed = np.random.randint(0, 1e9)
                ls_res.append(p.apply_async(func, (self.args, self.kwargs, seed)))

            pbar = tqdm(total=max_count)

            while True:
                time.sleep(0.01)
                cond = False
                for res in ls_res.copy():
                    if res.ready():
                        count += 1
                        pbar.update(1)
                        self.samples.append(res.get(0))
                        ls_res.remove(res)
                        cond = condition(self) if condition else False

                if count >= max_count or cond:
                    stop(ls_res)
                    break

            pbar.close()


    def run(self, condition=None):
        if condition:
            return self.run_until_condition(condition)

        self.samples = list()
        print(f"Run for {self.number} iterations, with {self.args} and {self.kwargs}")
        for i in range(self.number):
            if self.safe_mode:
                sample = self._safe_run_one()
            else:
                sample = self._run_one()

            if sample:
                self.samples.append(sample)

    def run_until_condition(self, condition):
        """
        :param condition: is a callback that take self as argument
        and return a boolean, True if condition is fulfilled and
        the loop can stop.
        The loop will always stop if max_number iterations is reached
        """
        self.samples = list()
        print(f"Run for max {self.max_number} iterations, with {self.args} and {self.kwargs}")
        for i in range(self.max_number):
            if self.safe_mode:
                sample = self._safe_run_one()
            else:
                sample = self._run_one()

            if sample:
                self.samples.append(sample)

            if condition(self):
                return

    def _run_one(self):
        sample = self.cls_(*self.args, **self.kwargs)
        sample.run()
        return sample

    def _safe_run_one(self):
        trial = 0
        while trial < 5:
            trial += 1
            try:
                sample = self.cls_(*self.args, **self.kwargs)
                sample.run()
            except Exception as e:
                print("Missed trial", trial, e)
            else:
                return sample
        raise RuntimeError("Could not get the cls_ to work after 5 trials")

    def stats(self, name, fillna=True, fillval=0):
        ret = np.concatenate([sample.stats[name] for sample in self.samples])
        if fillna:
            return np.nan_to_num(ret, copy=True, nan=fillval, posinf=None, neginf=None)

        return ret

    def show(self, ls):
        """
        ls must be a list of tuple with the form (var_name, label_expression)
        """
        ref = self.stats("time")
        plt.figure(figsize=(10, 10))
        for var, label in ls:
            sns.lineplot(x=ref, y=self.stats(var), label=label)
        plt.legend()

    def get_final_dist(self, name):
        return [sample.stats[name].iloc[-1] for sample in self.samples]

    def get_final_std(self, name, mean=False):
        std = np.std([sample.stats[name].iloc[-1] for sample in self.samples])
        if mean:
            return std / np.sqrt(len(self.samples))
        else:
            return std

    def get_final(self, name):
        return np.mean([sample.stats[name].iloc[-1] for sample in self.samples])

    def get_final_ratio_dist(self):
        """
        This function is only valid for bi state and not for tri state
        or other kinds of model
        """
        return [
                sample.stats["size_type_IP"].iloc[-1]
                / (
                    sample.stats["size_type_RG"].iloc[-1]
                    + sample.stats["size_type_IP"].iloc[-1]
                )
                for sample in self.samples
            ]

    def get_final_ratio(self):
        return np.mean(self.get_final_ratio_dist())

    def get_final_ratio_std(self):
        return np.std(self.get_final_ratio_dist())

    def show_pop(self):
        ls = [
            ("progenitor_pop_size", "Progenitor population"),
            ("whole_pop_size", "Whole Population"),
        ]
        self.show(ls)

    def show_progenitor_pop(self):
        ls = [("size_type_RG", "RG number"), ("size_type_IP", "IP number")]
        self.show(ls)

    def show_ratio_IP(self, ref_ratio):
        ref = self.stats("time")
        plt.figure(figsize=(10, 10))
        rg, ip = self.stats("size_type_RG"), self.stats("size_type_IP")
        # gp = self.stats("size_type_GP")
        # ratio = ip / (rg + ip + gp)
        ratio = ip / (rg + ip)
        sns.lineplot(x=ref, y=ratio, label="IP ratio")
        plt.plot(ref_ratio.index, ref_ratio.val / 100, label="Reference IP ratio")
        plt.legend()
