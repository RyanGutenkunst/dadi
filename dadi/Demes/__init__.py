import numpy as np
from . import Demes, Inference, DemesUtil

from .Demes import SFS

# Format for Cache: 
# Each event corresponds to a call to an integration or other manipulative method
# Integration methods record the time interval and population sizes. For the non-const methods, this is a list of values.
# For splits, record ancestry proportions 
cache = []

class Event:
    def __init__(self, duration=0, deme_ids=None):
        self.duration = duration
        self.end_time = None
        self.deme_ids = None
        if deme_ids is not None:
            self.deme_ids = tuple(deme_ids)

class Initiation(Event):
    def __init__(self, nu, deme_ids=None):
        super().__init__(np.inf, deme_ids=deme_ids)
        self.start_sizes = [nu]
        self.end_sizes = None

class Integration(Event):
    def __init__(self, duration, deme_ids):
        super().__init__(duration, deme_ids=deme_ids)

class IntegrationConst(Integration):
    def __init__(self, duration, start_sizes, mig=[], deme_ids=None):
        super().__init__(duration, deme_ids=deme_ids)
        self.start_sizes = start_sizes
        self.end_sizes = None
        self.mig = mig

class IntegrationNonConst(Integration):
    def __init__(self, history, deme_ids=None):
        super().__init__(duration = history[-1][0] - history[0][0], deme_ids=deme_ids)
        self.start_sizes = history[0][1]
        self.end_sizes = history[-1][1]
        self.mig = history[0][2]
        self.history = history
        self.linear = self.check_linear()
    def check_linear(self):
        """
        For each pop, check whether size change was linear.

        Does so by checking steps at duration * (1/3, 1/2, and 2/3)
        """
        # To store result
        pop_linear = [True]*len(self.history[0][1])
        # Steps to check
        steps_to_check = [len(self.history)//2, len(self.history)//3, len(self.history)*2//3]

        start_time = self.history[0][0]
        for pop_ii in range(len(self.history[0][1])):
            start_size, final_size = self.history[0][1][pop_ii], self.history[-1][1][pop_ii]
            slope = (final_size - start_size)/self.duration
            for step_ii in steps_to_check:
                pred_size = start_size + slope * (self.history[step_ii][0] - start_time)
                if not np.allclose(pred_size, self.history[step_ii][1][pop_ii]):
                    pop_linear[pop_ii] = False
                    break
        return pop_linear

class Split(Event):
    def __init__(self, proportions, deme_ids=None):
        super().__init__(deme_ids=deme_ids)
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
    # If we don't have deme names
    if cache[0].deme_ids is None:
        cache[0].deme_ids = ['d1_1']
    era = 1
    for older, younger in zip(cache[:-1], cache[1:]):
        if younger.deme_ids is None:
            if isinstance(younger, Split):
                if older.duration == 0:
                    # older was itself a zero-duration structural event (most
                    # often another Split), so this Split is simultaneous with
                    # it -- no integration has occurred to give any continuing
                    # deme an epoch yet. In dadi, a Split only creates one new
                    # deme, always in the last position, so here we keep the
                    # existing identity of every other deme and only assign a
                    # fresh id to the new one. (Renaming all of them, as below,
                    # would orphan the continuing demes with zero total
                    # duration, which demes cannot represent: every epoch must
                    # have start_time > end_time.) This is what allows e.g.
                    # simultaneous allopolyploid subgenome formation, via two
                    # Splits in a row, to be output correctly.
                    era += 1
                    new_id = 'd{0}_{1}'.format(era, len(older.deme_ids)+1)
                    younger.deme_ids = list(older.deme_ids) + [new_id]
                else:
                    # older had nonzero duration, so every deme already has a
                    # valid epoch from it. Start a fresh era for all of them,
                    # as dadi has always done.
                    era += 1
                    younger.deme_ids = ['d{0}_{1}'.format(era, ii+1) for ii in range(len(older.deme_ids)+1)]
            elif isinstance(younger, Remove):
                younger.deme_ids = list(older.deme_ids)
                del younger.deme_ids[younger.removed-1]
                younger.deme_ids = tuple(younger.deme_ids)
            elif isinstance(younger, Reorder):
                younger.deme_ids = [older.deme_ids[_-1] for _ in younger.neworder]
            elif younger.duration > 0 and older.duration > 0:
                era += 1
                younger.deme_ids = ['d{0}_{1}'.format(era, ii+1) for ii in range(len(older.deme_ids))]
            else:
                younger.deme_ids = older.deme_ids

    # Substitute deme names
    if deme_mapping is not None:
        map = {}
        for newname, oldnames in deme_mapping.items():
            for oldname in oldnames:
                map[oldname] = newname
        for e in cache:
            e.deme_ids = [map.get(d, d) for d in e.deme_ids]

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
    deme_info = {}
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
                    if e.linear[d_ii]:
                        epochs[-1]['size_function'] = 'linear'
                if Nref is not None:
                    epochs[-1]['end_time'] *= 2*Nref
                    if generation_time is not None:
                        epochs[-1]['end_time'] *= generation_time
                    epochs[-1]['start_size'] *= Nref
                    if isinstance(e, IntegrationNonConst):
                        epochs[-1]['end_size'] *= Nref
                if ii > 0 and ancestors is None and deme not in cache[ii-1].deme_ids:
                    # If demes is new due to Integration
                    start_time = e.end_time + e.duration
                    if Nref is not None:
                        start_time *= 2*Nref
                        if generation_time is not None:
                            start_time *= generation_time
                    d_ii = e.deme_ids.index(deme)
                    ancestors = [cache[ii-1].deme_ids[d_ii]]
                    proportions = [1]
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

        deme_info[deme] = {'epochs':epochs, 'start_time':start_time,
                            'ancestors':ancestors, 'proportions':proportions}

    # A chain of two or more simultaneous Splits (zero duration between them,
    # see above) can leave a deme's resolved ancestor with the exact same
    # start_time as the deme itself -- e.g. a new deme created by the second
    # Split in the chain, whose ancestor is a deme that was itself just
    # (re)created by the first Split. demes requires an ancestor to already
    # exist when a deme branches off (a strictly earlier start_time), so we
    # walk back through any such same-instant ancestors to whichever earlier
    # deme(s) they themselves descend from, combining proportions along the
    # way.
    for deme in all_demes:
        info = deme_info[deme]
        if not info['ancestors']:
            continue
        resolved = {}
        pending = list(zip(info['ancestors'], info['proportions']))
        while pending:
            anc, prop = pending.pop()
            anc_info = deme_info[anc]
            if anc_info['start_time'] == info['start_time']:
                pending.extend((a, prop*p) for a, p in
                                zip(anc_info['ancestors'], anc_info['proportions']))
            else:
                resolved[anc] = resolved.get(anc, 0) + prop
        info['ancestors'] = list(resolved.keys())
        info['proportions'] = list(resolved.values())

    for deme in all_demes:
        info = deme_info[deme]
        b.add_deme(deme, epochs=info['epochs'], start_time=info['start_time'],
                   ancestors=info['ancestors'], proportions=info['proportions'])

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
            if Nref is not None:
                e.end_time *= 2*Nref
                if generation_time is not None:
                    e.end_time *= generation_time
            b.add_pulse(sources=sources, dest=dest, proportions = e.proportions, time=e.end_time)

    graph = b.resolve()

    return graph
