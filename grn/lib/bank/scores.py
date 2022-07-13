from lib.utils import normalize_time, align_time
from jf.autocompute.jf import O
from brain import BrainModel
from submodels import factories
from lib.callback import (
    cell_number_callback, progenitor_number_callback, neuron_number_callback,
    TargetPopulation, TagNumberCallback,
)


def score_linobj(bb, ref, *args, **kwargs):
    stats = bb.stats.copy()
    stats, ref_stats = align_time(stats, ref.stats)
    stats_p, ref_p = normalize_time(stats, ref_stats, "progenitor_pop_size", "progenitor_pop_size")
    stats_n, ref_n = normalize_time(stats, ref_stats, "neuron_pop_size", "neuron_pop_size",
                                    "progenitor_pop_size", "progenitor_pop_size")
    last_time_stats, last_time_ref = max(stats.time), max(ref_stats.time)
    
    ref_p = ref_p.set_index("time")
    ref_n = ref_n.set_index("time")
    stats_p = stats_p.set_index("time")
    stats_n = stats_n.set_index("time")
    
    prog = stats_p.loc[last_time_stats]["progenitor_pop_size"]
    neuron = stats_n.loc[last_time_stats]["neuron_pop_size"]
    
    ref_prog = ref_p.loc[last_time_ref]["progenitor_pop_size"]
    ref_neuron = ref_n.loc[last_time_ref]["neuron_pop_size"]

    return 1 / max(1, 1000 - abs(prog - ref_prog) - abs(neuron - ref_neuron))


def get_bb(prun, grn):
    ccls = factories[prun.model](grn=grn)
    callbacks = dict(
        progenitor_pop_size=progenitor_number_callback,
        whole_pop_size=cell_number_callback,
        neuron_pop_size=neuron_number_callback,
    )
    bb = BrainModel(time_step=prun.ts, verbose=False, start_population=prun.size,
                    max_pop_size=prun.max_cell_pop_size, cell_cls=ccls,
                    end_time=prun.end_time, start_time=prun.start_time, silent=True, opti=True,
              run_tissue=True, monitor_callbacks=callbacks)
    return bb


def fitness_multistep(prun, grn, steps):
    total_fitness = 0
    stop = False
    previous_time = None
    bb = get_bb(prun, grn)
    # first step
    for step in steps:
        if not bb.run_until(step.end_time):
            stop = True
        
        score_step = step.score_func(bb, prun.ref, max_step=step.end_time, min_step=step.start_time)
        fitness_step = 1.0 / score_step
        fitness_step = min(fitness_step, step.max_fitness)
        total_fitness += fitness_step
        if fitness_step < step.min_fitness or stop:
            return total_fitness, bb.stats
        else:
            previous_time = step.end_time
            step.passed()
        
    return total_fitness, bb.stats


class ObjectiveStep(O):
    start_time = 0
    end_time = 0
    max_fitness = 1e9
    min_fitness = 1
    name = ""
    _passed = False
    
    def reset(self):
        self._passed = False
    
    def passed(self):
        pass