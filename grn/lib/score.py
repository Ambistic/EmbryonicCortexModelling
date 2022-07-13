import pandas as pd
import numpy as np
from collections import Counter

from lib.preprocess import preprocess_whole_size, preprocess_progenitor_size
from lib.utils import (shrink_and_align_stats, align_time, normalize_time,
                       shrink_time, as_time_lists)

REF = pd.read_csv("reference/ref_tristate2.csv")  # ref is a mean


def score_coefficient_variation_old(x, y):
    return sum([(x_ - y_)**2 / x_ for x_, y_ in zip(x, y)])


def score_coefficient_variation(x, y):
    return sum([(x_ - y_)**2 / np.sqrt(np.abs(x_)) for x_, y_ in zip(x, y)])


def score_coefficient_variation_norm(x, y, norm=1):
    return sum([(x_ - y_)**2 / max(1, np.abs(x_)**norm) for x_, y_ in zip(x, y)])


def score_progenitor_size(stats, ref=REF, max_step=None, min_step=None):
    stats, ref = shrink_and_align_stats(stats, ref, max_step, min_step)
    x, y = preprocess_progenitor_size(stats, ref)
    return score_coefficient_variation(x, y)


def score_whole_size(stats, ref=REF, max_step=None, min_step=None):
    stats, ref = shrink_and_align_stats(stats, ref, max_step, min_step)
    x, y = preprocess_whole_size(stats, ref)
    return score_coefficient_variation(x, y)


def score_both_size(stats, ref=REF, max_step=None, min_step=None):
    return score_progenitor_size(stats, ref, max_step, min_step) + score_whole_size(stats, ref, max_step, min_step)


def score_progenitor_size_new(stats, ref=REF, max_step=None, min_step=None):
    colname = "progenitor_pop_size"
    stats, ref = align_time(stats, ref)
    stats, ref = normalize_time(stats, ref, col_data=colname, col_ref=colname)
    stats, ref = shrink_time(stats, ref, min_step, max_step)
    x, y = as_time_lists(stats, ref, col_data=colname, col_ref=colname)
    
    return score_coefficient_variation(x, y)


def score_whole_size_new(stats, ref=REF, max_step=None, min_step=None):
    colname = "whole_pop_size"
    stats, ref = align_time(stats, ref)
    stats, ref = normalize_time(stats, ref, col_data=colname, col_ref=colname)
    stats, ref = shrink_time(stats, ref, min_step, max_step)
    x, y = as_time_lists(stats, ref, col_data=colname, col_ref=colname)
    
    return score_coefficient_variation(x, y)


def score_both_size_new(stats, ref=REF, max_step=None, min_step=None):
    return score_progenitor_size_new(stats, ref, max_step, min_step) + score_whole_size_new(stats, ref, max_step, min_step)


def score_progenitor_size_norm(stats, ref=REF, max_step=None, min_step=None, norm=1):
    colname = "progenitor_pop_size"
    stats, ref = align_time(stats, ref)
    stats, ref = normalize_time(stats, ref, col_data=colname, col_ref=colname)
    stats, ref = shrink_time(stats, ref, min_step, max_step)
    x, y = as_time_lists(stats, ref, col_data=colname, col_ref=colname)
    
    return score_coefficient_variation_norm(x, y, norm=norm)


def score_whole_size_norm(stats, ref=REF, max_step=None, min_step=None, norm=1):
    colname = "whole_pop_size"
    stats, ref = align_time(stats, ref)
    stats, ref = normalize_time(stats, ref, col_data=colname, col_ref=colname)
    stats, ref = shrink_time(stats, ref, min_step, max_step)
    x, y = as_time_lists(stats, ref, col_data=colname, col_ref=colname)
    
    return score_coefficient_variation_norm(x, y, norm=norm)


def score_both_size_norm(stats, ref=REF, max_step=None, min_step=None, norm=1):
    return score_progenitor_size_norm(stats, ref, max_step, min_step, norm=norm) \
        + score_whole_size_norm(stats, ref, max_step, min_step, norm=norm)


def score_stats_norm(stats, ref, col_stats, col_ref, max_step=None, min_step=None,
                     norm=1, col_norm_data=None, col_norm_ref=None):
    stats, ref = align_time(stats, ref)
    stats, ref = normalize_time(stats, ref, col_data=col_stats, col_ref=col_ref,
                               col_norm_data=col_norm_data, col_norm_ref=col_norm_ref)
    stats, ref = shrink_time(stats, ref, min_step, max_step)
    x, y = as_time_lists(stats, ref, col_data=col_stats, col_ref=col_ref)
    
    return score_coefficient_variation_norm(x, y, norm=norm)


def score_stats_norm_rel(stats, ref, col_stats, col_ref, max_step=None, min_step=None,
                     norm=1, col_norm_data=None, col_norm_ref=None):
    stats, ref = align_time(stats, ref)
    stats, ref = shrink_time(stats, ref, min_step, max_step)
    stats, ref = normalize_time(stats, ref, col_data=col_stats, col_ref=col_ref,
                               col_norm_data=col_norm_data, col_norm_ref=col_norm_ref)
    x, y = as_time_lists(stats, ref, col_data=col_stats, col_ref=col_ref)
    
    return score_coefficient_variation_norm(x, y, norm=norm)


def fate_corr(cell_pairs, kC="Progenitor", kN="PostMitotic"):
    counts = Counter(cell_pairs)
    
    CC, CN, NN = counts[(kC, kC)], counts[(kN, kC)] + counts[(kC, kN)], counts[(kN, kN)]
    T = CC + CN + NN
    pCC, pCN, pNN = CC / T, CN / T, NN / T
    all_C, all_N = 2 * CC + CN, 2 * NN + CN
    pC, pN = all_C / (T * 2), all_N / (T * 2)
    eCC, eCN, eNN = pC**2, 2*pC*pN, pN**2
    f_metric = 1 - pCN / eCN
    return f_metric
