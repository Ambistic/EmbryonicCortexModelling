import pandas as pd
from ete3 import Tree, NodeStyle
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path as P
from collections import defaultdict, Counter


style_RG = NodeStyle()
style_RG["fgcolor"] = "purple"
style_RG["size"] = 15

style_IP = NodeStyle()
style_IP["fgcolor"] = "green"
style_IP["size"] = 15

style_N = NodeStyle()
style_N["fgcolor"] = "red"
style_N["size"] = 15

style_GP = NodeStyle()
style_GP["fgcolor"] = "yellow"
style_GP["size"] = 15

STYLE = defaultdict(NodeStyle, RG=style_RG, IP=style_IP, PostMitotic=style_N, GP=style_GP)

DEFAULT_PARAMS = [
    ("bias_ratio_1", 0.2),
    ("bias_ratio_2", 0.15),
    ("bias_ratio_3", -0.1),
    ("bias_ratio_4", -0.1),
    ("bias_ratio_5", -0.1),

    ("tc_coeff_1", 1.),
    ("tc_coeff_2", 1.),
    ("tc_coeff_3", 1.),
    ("tc_coeff_4", 1.),
    ("tc_coeff_5", 1.),

    ("tc_coeff_RG_1", 1.),
    ("tc_coeff_RG_2", 1.),
    ("tc_coeff_RG_3", 1.),
    ("tc_coeff_RG_4", 1.),
    ("tc_coeff_RG_5", 1.),

    ("tc_coeff_IP_1", 1.),
    ("tc_coeff_IP_2", 1.),
    ("tc_coeff_IP_3", 1.),
    ("tc_coeff_IP_4", 1.),
    ("tc_coeff_IP_5", 1.),

    ("diff_values_1", 0.73),
    ("diff_values_2", 0.63),
    ("diff_values_3", 0.47),
    ("diff_values_4", 0.45),
    ("diff_values_5", 0.45),

    ("diff_values_IP_1", 0.23),
    ("diff_values_IP_2", 0.23),
    ("diff_values_IP_3", 0.23),
    ("diff_values_IP_4", 0.23),
    ("diff_values_IP_5", 0.23),

    ("diff_values_RG_1", 0.63),
    ("diff_values_RG_2", 0.53),
    ("diff_values_RG_3", 0.43),
    ("diff_values_RG_4", 0.38),
    ("diff_values_RG_5", 0.33),

    ("diff_values_RG_IP_1", 0.63),
    ("diff_values_RG_IP_2", 0.53),
    ("diff_values_RG_IP_3", 0.43),
    ("diff_values_RG_IP_4", 0.38),
    ("diff_values_RG_IP_5", 0.33),

    ("diff_values_RG_GP_1", 1.),
    ("diff_values_RG_GP_2", 1.),
    ("diff_values_RG_GP_3", 1.),
    ("diff_values_RG_GP_4", 0.8),
    ("diff_values_RG_GP_5", 0.6),  
]

def get_default_val(name):
    defaults = {k: v for k, v in DEFAULT_PARAMS}
    return defaults.get(name.strip("_"))


defaults_metrics_mean = dict(
    corr_tc_output_g3 = 0.09,
    corr_tc_output_g23 = 0.19,
    corr_tc_output_g2 = 0.09,
    corr_tc_daughter_g3 = 0.565,
    corr_tc_daughter_g23 = 0.68,
    corr_tc_daughter_g2 = 0.695,
    F_metric = 0.397,
    mean_ratio_m_d_g23 = 1.07,
    mean_diff_m_d_g23 = -0.0008,
    std_ratio_m_d_g23 = 0.426,
    std_diff_m_d_g23 = 19.095,
)

defaults_metrics_sd = dict(
    corr_tc_output_g3 = 0.14,
    corr_tc_output_g23 = 0.11,
    corr_tc_output_g2 = 0.20,
    corr_tc_daughter_g3 = 0.13,
    corr_tc_daughter_g23 = 0.09,
    corr_tc_daughter_g2 = 0.16,
    F_metric = 0,
    mean_ratio_m_d_g23 = 0,
    mean_diff_m_d_g23 = 0,
    std_ratio_m_d_g23 = 0,
    std_diff_m_d_g23 = 0,
)

def get_default_metric_mean(k):
    return defaults_metrics_mean.get(k, 0)

def get_default_metric_sd(k):
    return defaults_metrics_sd.get(k, 0)