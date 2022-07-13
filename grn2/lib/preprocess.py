def preprocess_progenitor_size(stats, ref):
    colname_ref = "progenitor_pop_size"
    colname_stats = "progenitor_pop_size"
    
    min_step = min(ref.index)  # ref.index is the time
    stats[colname_stats] = stats[colname_stats] / stats[colname_stats].get(min_step)
    ref[colname_ref] = ref[colname_ref] / ref[colname_ref].get(min_step)
    x = [ref[colname_ref].get(t, 0) for t in ref.index]
    y = [stats[colname_stats].get(t, 0) for t in ref.index]
    return x, y


def preprocess_whole_size(stats, ref):
    colname_ref = "whole_pop_size"
    colname_stats = "whole_pop_size"
    
    min_step = min(ref.index)  # ref.index is the time
    stats[colname_stats] = stats[colname_stats] / stats[colname_stats].get(min_step)
    ref[colname_ref] = ref[colname_ref] / ref[colname_ref].get(min_step)
    x = [ref[colname_ref].get(t, 0) for t in ref.index]
    y = [stats[colname_stats].get(t, 0) for t in ref.index]
    return x, y


def preprocess_progenitor_size_2(stats, ref):
    colname_ref = "progenitor_pop_size"
    colname_stats = "progenitor_pop_size"
    
    min_step = min(ref.index)  # ref.index is the time
    stats[colname_stats] = stats[colname_stats] / stats[colname_stats].get(min_step)
    ref[colname_ref] = ref[colname_ref] / ref[colname_ref].get(min_step)
    x = [ref[colname_ref].get(t, 0) for t in ref.index]
    y = [stats[colname_stats].get(t, 0) for t in stats.index]
    return x, y


def preprocess_whole_size_2(stats, ref):
    colname_ref = "whole_pop_size"
    colname_stats = "whole_pop_size"
    
    min_step = min(ref.index)  # ref.index is the time
    stats[colname_stats] = stats[colname_stats] / stats[colname_stats].get(min_step)
    ref[colname_ref] = ref[colname_ref] / ref[colname_ref].get(min_step)
    x = [ref[colname_ref].get(t, 0) for t in ref.index]
    y = [stats[colname_stats].get(t, 0) for t in stats.index]
    return x, y


def all_daughter_defined(cell, population):
    """ Warning : the word Cell must correspond to an unknown type"""
    if not len(cell.children) == 2:
        return False
    
    for daughter_id in cell.children:
        if population[daughter_id].type().name == "Cell":
            return False
    return True


def pairs(cells, population):
    return [tuple(population[x].type().name for x in cell.children) for cell in cells]


def get_fmetric_pairs(population, min_time=50, max_time=60):
    """ This function is only valid for GRNCells"""
    cells_in_window = [c for c in population.values() 
                       if c.appear_time >= min_time and c.appear_time < max_time]
    cells_divided = [c for c in cells_in_window if c.type().name == "Progenitor"]
    final_cells = list(filter(lambda x: all_daughter_defined(x, population), cells_divided))
    
    return pairs(final_cells, population)