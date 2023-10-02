from . import Demes, Inference

from .Demes import SFS

# Format for Cache: 
# Each event corresponds to a call to an integration or other manipulative method
# Integration methods record the time interval and population sizes. For the non-const methods, this is a list of values.
# For splits, record ancestry proportions 
cache = []

class Event:
    def __init__(self, duration=0):
        self.duration = duration
        self.end_time = None
        self.deme_ids = None

class Initiation(Event):
    def __init__(self, nu):
        super().__init__(np.inf)
        self.start_sizes = [nu]
        self.end_sizes = None

class Integration(Event):
    def __init__(self, duration):
        super().__init__(duration)

class IntegrationConst(Integration):
    def __init__(self, duration, start_sizes, mig=[]):
        super().__init__(duration)
        self.start_sizes = start_sizes
        self.end_sizes = None
        self.mig = mig

class IntegrationNonConst(Integration):
    def __init__(self, history):
        super().__init__(duration = history[-1][0] - history[0][0])
        self.start_sizes = history[0][1]
        self.end_sizes = history[-1][1]
        self.mig = history[0][2]
        self.history = history

class Split(Event):
    def __init__(self, proportions):
        super().__init__()
        self.proportions = proportions

class Remove(Event):
    def __init__(self, removed):
        super().__init__()
        self.removed = removed 

class Reorder(Event):
    def __init__(self, neworder):
        super().__init__()
        self.neworder = neworder

class Pulse(Event):
    def __init__(self, sources, dest, proportions):
        super().__init__()
        self.dest = dest
        # Filter out zeros
        self.sources = [sources[_] for _ in range(len(proportions)) if proportions[_] != 0]
        self.proportions= [proportions[_] for _ in range(len(proportions)) if proportions[_] != 0]

import numpy as np
import demes
def output(Nref=None, deme_mapping=None, generation_time=None):
    """
    Note: If no Nref is specified, then migration rates are scaled to lie within 0 to 1, which is required by the demes specification.
    """
    global cache

    # Proceed from present to past to get e end_times
    cache[-1].end_time = 0 # Last e ends at present time
    for younger, older in zip(cache[::-1][:-1], cache[::-1][1:]):
        older.end_time = younger.end_time + younger.duration

    # XXX: Should I number demes as d{pop_number}.{era}?

    # Create and propagate names for all demes, starting from d0
    cache[0].deme_ids, era = ['d1_1'], 1
    for older, younger in zip(cache[:-1], cache[1:]):
        if younger.deme_ids is None:
            if isinstance(younger, Split):
                era += 1
                younger.deme_ids = ['d{0}_{1}'.format(ii+1, era) for ii in range(len(older.deme_ids)+1)]
            elif isinstance(younger, Remove):
                younger.deme_ids = older.deme_ids[:]
                del younger.deme_ids[younger.removed-1]
            elif isinstance(younger, Reorder):
                younger.deme_ids = [older.deme_ids[_-1] for _ in younger.neworder]
            else:
                younger.deme_ids = older.deme_ids

    # Substitute deme names
    if deme_mapping is not None:
        map = {}
        for newname, oldnames in deme_mapping.items():
            for oldname in oldnames:
                map[oldname] = newname
        for e in cache:
            e.deme_ids = [map[d] for d in e.deme_ids]

    # Collect all demes in the history, in order from oldest to newest.
    all_demes = []
    for e in cache:
        for p in e.deme_ids:
            if p not in all_demes:
                all_demes.append(p)

    if Nref is None:
        b = demes.Builder(time_units='scaled', generation_time=1)
    else:
        if generation_time is None:
            b = demes.Builder(time_units='generations')
        else:
            b = demes.Builder(time_units='years', generation_time=generation_time)

    # Build up info for each deme
    for deme in all_demes:
        epochs = []
        start_time, ancestors, proportions = None, None, None
        for ii, e in enumerate(cache):
            if deme not in e.deme_ids:
                continue
            # Index of this deme in this events's deme_ids list
            d_ii = e.deme_ids.index(deme)
            if isinstance(e, Initiation) or isinstance(e, Integration):
                epochs.append({'end_time':e.end_time, 'start_size':e.start_sizes[d_ii]})
                if isinstance(e, IntegrationNonConst):
                    epochs[-1]['end_size'] = e.end_sizes[d_ii]
                if Nref is not None:
                    epochs[-1]['end_time'] *= 2*Nref
                    if generation_time is not None:
                        epochs[-1]['end_time'] *= generation_time
                    epochs[-1]['start_size'] *= Nref
                    if isinstance(e, IntegrationNonConst):
                        epochs[-1]['end_size'] *= Nref
            if isinstance(e, Split): 
                prev_e = cache[ii-1]
                if deme not in prev_e.deme_ids: # If new deme
                    start_time = e.end_time
                    if Nref is not None:
                        start_time *= 2*Nref
                        if generation_time is not None:
                            start_time *= generation_time
                    # In dadi, the newly created pop in a Split is always the last one.
                    if d_ii != len(e.deme_ids)-1:
                        # So the others are simply ancestral to the old demes
                        ancestors = [prev_e.deme_ids[d_ii]]
                        proportions = [1]
                    else:
                        # For the new pop, ancestors are those with non-zero contribution
                        ancestors = [prev_e.deme_ids[_] for _ in range(len(prev_e.deme_ids))
                                     if e.proportions[_] != 0]
                        proportions = [e.proportions[_] for _ in range(len(prev_e.deme_ids))
                                     if e.proportions[_] != 0]

        b.add_deme(deme, epochs=epochs, start_time=start_time, ancestors=ancestors, proportions=proportions)

    all_migs = []
    for e in cache:
        if isinstance(e, Integration):
            start_time = e.end_time + e.duration
            m_ii = 0
            for dest in e.deme_ids:
                for source in e.deme_ids:
                    if dest == source:
                        continue
                    if e.mig[m_ii] != 0:
                        all_migs.append({'rate':e.mig[m_ii], 'source':source, 'dest':dest,
                                         'start_time':start_time, 'end_time':e.end_time})
                    m_ii += 1

    if Nref is not None:
        for m in all_migs:
            m['rate'] /= 2*Nref
            m['start_time'] *= 2*Nref
            m['end_time'] *= 2*Nref
            if generation_time is not None:
                m['start_time'] *= generation_time
                m['end_time'] *= generation_time
    else: # Normalize migrations by total influx, to ensure no total influx exceeds 1
        max_in_mig = 0
        for d in all_demes:
            tot_mig = np.sum([_['rate'] for _ in all_migs if _['dest'] == d])
            max_in_mig = max(max_in_mig, tot_mig)
        for m in all_migs:
            m['rate'] /= max_in_mig

    for m in all_migs:
        b.add_migration(**m)

    # Add pulses of migration
    for e in cache:
        if isinstance(e, Pulse) and len(e.sources) > 0:
            sources = [e.deme_ids[ii-1] for ii in e.sources]
            dest = e.deme_ids[e.dest-1]
            b.add_pulse(sources=sources, dest=dest, proportions = e.proportions, time=e.end_time)

    graph = b.resolve()

    return graph