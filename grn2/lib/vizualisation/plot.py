import matplotlib.pyplot as plt
from matplotlib import cm

from .preprocess import (
    preprocess_nb_progenitor, preprocess_nb_cells, preprocess_ratio,
    preprocess_progeny, preprocess_progeny_all, preprocess_corr_metrics,
    mean_sd, mean_dict, mean_progeny
)

from .defaults import get_default_metric_mean, get_default_metric_sd

from .biodata import plot_ratio_eomes, plot_number_cells

class RendererPlot:
    def __init__(self, name, preprocess_func, mean_func, plot_func):
        self.name = name
        self.preprocess_func = preprocess_func
        self.mean_func = mean_func
        self.plot_func = plot_func
        
    def __call__(self, metagroup, root, title=None):
        lines = list()
        for group_value, group in metagroup.items():  # different parameters
            ls = []
            for sample in group:  # different trials, only random init changes
                ls.append(self.preprocess_func(sample, root))
            line = self.mean_func(ls)
            line["label"] = group_value
            lines.append(line)
        self.plot_func(lines, title=title)
        
    # no preprocess here !
    def make_one(self, ls):
        line = self.mean_func(ls)
        self.plot_func(line)
        
    def make_from_datalist(self, datalist, dataname, title=None):
        ls = []
        for data in datalist:
            ls.append(self.preprocess_func(**{dataname: data}))
        line = self.mean_func(ls)
        self.plot_func([line], title=title)
        
        
def draw_sum(line):
    total = 0
    for k in sorted(set(line.keys()) - {"x", "label"}):
        plt.fill_between(line["x"], total, total + line[k]["mean"], label=k)
        total = total + line[k]["mean"]
    plt.legend()

def draw_line(x, y, label=None):
    if isinstance(y, dict):
        return draw_mean_sd_line(x, y["mean"], y["sd"], label=label)
        
    plt.plot(x, y, label=label)
        
def draw_mean_sd_line(x, mean, sd, label=None):
    plt.plot(x, mean, label=label)
    plt.fill_between(x, (mean-sd), (mean+sd), alpha=.3)

def plot_lines(lines, title=None):
    lines.sort(key=lambda x: x.get("label", "NO_LABEL"))
    plt.figure()
    for line in lines:
        draw_line(line["x"], line["y"], line.get("label"))
    plt.legend()
    if title:
        plt.title(title)
        
def plot_lines_ratio_eomes(lines, title=None):
    lines.sort(key=lambda x: x.get("label", "NO_LABEL"))
    plt.figure()
    for line in lines:
        draw_line(line["x"], line["y"], line.get("label"))
    plot_ratio_eomes()
    plt.legend()
    if title:
        plt.title(title)
        
def plot_lines_number_cells(lines, title=None):
    lines.sort(key=lambda x: x.get("label", "NO_LABEL"))
    plt.figure()
    for line in lines:
        draw_line(line["x"], line["y"], line.get("label"))
    plot_number_cells()
    plt.legend()
    if title:
        plt.title(title)

def plot_sum(lines, title=None):
    lines.sort(key=lambda x: x.get("label", "NO_LABEL"))
    if len(lines) > 6:
        raise ValueError("Maximum allowed plot is 6")
    plt.figure(figsize=(10, 12))
    for i, line in enumerate(lines):
        plt.subplot(3, 2, i + 1)
        draw_sum(line)
        plt.title(line.get("label", "NO_LABEL"))
    if title:
        plt.suptitle(title)
        
def plot_metrics(lines, title=None):
    plt.figure(figsize=(16, 12))
    keys = set().union(*[element.keys() for element in lines]) - {"label"}
    lines.sort(key=lambda x: x.get("label", "NO_LABEL"))
    labels = [element.get("label", "NO_LABEL") for i, element in enumerate(lines)]
    null = {"mean": 0, "sd": 0}
    for i, k in enumerate(keys):
        plt.subplot(3, 4, i + 1)
        plt.title(k)
        ref_label, ref_vec, ref_err = "Ref TLV", get_default_metric_mean(k), get_default_metric_sd(k)
        vec = [element.get(k, null).get("mean") for element in lines]
        err = [element.get(k, null).get("sd") for element in lines]
        plt.bar([ref_label] + labels, 
                [ref_vec] + vec, 
                yerr=[ref_err] + err, color=cm.Set2.colors)
        
    if title:
        plt.suptitle(title)

render_nb_progenitor = RendererPlot("nb_progenitor", preprocess_nb_progenitor, mean_sd, plot_lines_number_cells)
render_nb_cells = RendererPlot("nb_cells", preprocess_nb_cells, mean_sd, plot_lines)
render_ratio = RendererPlot("ratioIP", preprocess_ratio, mean_sd, plot_lines_ratio_eomes)
render_progeny = RendererPlot("progeny leaves", preprocess_progeny, mean_progeny, plot_sum)
render_progeny_all = RendererPlot("progeny all", preprocess_progeny_all, mean_progeny, plot_sum)
render_corr_metrics = RendererPlot("corr_metrics", preprocess_corr_metrics, mean_dict, plot_metrics)
# render_neighborhood = RendererPlot(preprocess_neighborhood, mean_dict, plot_metrics)

renderers = [render_nb_progenitor, render_nb_cells, render_ratio,
            render_progeny, render_progeny_all, render_corr_metrics]