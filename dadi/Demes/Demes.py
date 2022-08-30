# This script contains functions to compute the sample SFS from a demography
# defined using demes. Not ever deme graph will be supported, as dadi can
# only handle integrating up to five populations.
# (and cannot include selfing or cloning?)

from collections import defaultdict
import math
import copy
import numpy as np

import dadi
import pickle

try:
    import demes
    _imported_demes = True
except ImportError:
    _imported_demes = False


def _check_demes_imported():
    if not _imported_demes:
        raise ImportError(
            "To simulate using demes, it must be installed -- "
            "try `pip install demes`"
        )


def SFS(g, sampled_demes, sample_sizes, sample_times=None, Ne=None, pts=None):
    """
    Takes a deme graph and computes the SFS. ``demes`` is a package for
    specifying demographic models in a user-friendly, human-readable YAML
    format. This function automatically parses the demographic description
    and returns a SFS for the specified populations and sample sizes.

    :param g: A ``demes`` DemeGraph from which to compute the SFS.
    :type g: :class:`demes.DemeGraph`
    :param sampled_demes: A list of deme IDs to take samples from. We can repeat
        demes, as long as the sampling of repeated deme IDs occurs at distinct
        times.
    :type sampled_demes: list of strings
    :param sample_sizes: A list of the same length as ``sampled_demes``,
        giving the sample sizes for each sampled deme.
    :type sample_sizes: list of ints
    :param sample_times: If None, assumes all sampling occurs at the end of the
        existence of the sampled deme. If there are
        ancient samples, ``sample_times`` must be a list of same length as
        ``sampled_demes``, giving the sampling times for each sampled
        deme. Sampling times are given in time units of the original deme graph,
        so might not necessarily be generations (e.g. if ``g.time_units`` is years)
    :type sample_times: list of floats, optional
    :param Ne: reference population size. If none is given, we use the initial
        size of the root deme.
    :type Ne: float, optional
    :return: A ``dadi`` site frequency spectrum, with dimension equal to the
        length of ``sampled_demes``, and shape equal to ``sample_sizes`` plus one
        in each dimension, indexing the allele frequency in each deme from 0
        to n[i], where i is the deme index.
    :rtype: :class:`dadi.Spectrum`
    """
    _check_demes_imported()
    if len(sampled_demes) != len(sample_sizes):
        raise ValueError("sampled_demes and sample_sizes must be same length")
    if sample_times is not None and len(sampled_demes) != len(sample_times):
        raise ValueError("sample_times must have same length as sampled_demes")
    for deme in sampled_demes:
        if deme not in g:
            raise ValueError(f"deme {deme} is not in demography")

    if sample_times is None:
        sample_times = [g[d].end_time for d in sampled_demes]

    if pts == None:
        raise ValueError("dadi requires defining pts (grid points)")

    # for any ancient samples, we need to add frozen branches
    # with this, all "sample times" are at time 0, and ancient sampled demes are frozen
    if np.any(np.array(sample_times) != 0):
        g, sampled_demes, list_of_frozen_demes = _augment_with_ancient_samples(
            g, sampled_demes, sample_times
        )
        sample_times = [0 for _ in sample_times]
    else:
        list_of_frozen_demes = []

    if g.time_units != "generations":
        g, sample_times = _convert_to_generations(g, sample_times)
    for d, n, t in zip(sampled_demes, sample_sizes, sample_times):
        if t < g[d].end_time or t >= g[d].start_time:
            raise ValueError("sample time for {deme} must be within its time span")

    # get the list of demographic events from demes, which is a dictionary with
    # lists of splits, admixtures, mergers, branches, and pulses
    demes_demo_events = g.discrete_demographic_events()

    # get the dict of events and event times that partition integration epochs, in
    # descending order. events include demographic events, such as splits and
    # mergers and admixtures, as well as changes in population sizes or migration
    # rates that require instantaneous changes in the size function or migration matrix.
    # get the list of demes present in each epoch, as a dictionary with non-overlapping
    # adjoint epoch time intervals
    demo_events, demes_present = _get_demographic_events(
        g, demes_demo_events, sampled_demes
    )

    for epoch, epoch_demes in demes_present.items():
        if len(epoch_demes) > 5:
            raise ValueError(
                f"dadi cannot integrate more than five demes at a time. "
                f"Epoch {epoch} has demes {epoch_demes}."
            )

    # get the list of size functions, migration matrices, and frozen attributes from
    # the deme graph and event times, matching the integration times
    nu_funcs, mig_mats, Ts, frozen_pops = _get_integration_parameters(
        g, demes_present, list_of_frozen_demes, Ne=Ne
    )

    # compute the SFS
    fs = _compute_sfs(
        demo_events,
        demes_present,
        sample_sizes,
        nu_funcs,
        mig_mats,
        Ts,
        frozen_pops,
        pts
    )


    return fs


##
## general functions used by both SFS
##


def _convert_to_generations(g, sample_times):
    """
    Takes a deme graph that is not in time units of generations and converts
    times to generations, using the time units and generation times given.
    """
    if g.time_units == "generations":
        return g, sample_times
    else:
        for ii, sample_time in enumerate(sample_times):
            sample_times[ii] = sample_time / g.generation_time
        g = g.in_generations()
        return g, sample_times


def _augment_with_ancient_samples(g, sampled_demes, sample_times):
    """
    Returns a demography object and new sampled demes where we add
    a branch event for the new sampled deme that is frozen.

    New sampled, frozen demes are labeled "{deme}_sampled_{sample_time}".
    Note that we cannot have multiple ancient sampling events at the same
    time for the same deme (for additional samples at the same time, increase
    the sample size).
    """
    frozen_demes = []
    b = demes.Builder.fromdict(g.asdict())
    for ii, (sd, st) in enumerate(zip(sampled_demes, sample_times)):
        if st > 0:
            sd_frozen = sd + f"_sampled_{st}"
            frozen_demes.append(sd_frozen)
            sampled_demes[ii] = sd_frozen
            b.add_deme(
                sd_frozen,
                start_time=st,
                epochs=[dict(end_time=0, start_size=1)],
                ancestors=[sd],
            )
    g = b.resolve()
    return g, sampled_demes, frozen_demes


def _get_demographic_events(g, demes_demo_events, sampled_demes):
    """
    Returns demographic events and present demes over each epoch.
    Epochs are divided by any demographic event.
    """
    # first get set of all time dividers, from demographic events, migration
    # rate changes, deme epoch changes
    break_points = set()
    for deme in g.demes:
        for e in deme.epochs:
            break_points.add(e.start_time)
            break_points.add(e.end_time)
    for pulse in g.pulses:
        break_points.add(pulse.time)
    for migration in g.migrations:
        break_points.add(migration.start_time)
        break_points.add(migration.end_time)

    # get demes present for each integration epoch
    integration_times = [
        (start_time, end_time)
        for start_time, end_time in zip(
            sorted(list(break_points))[-1:0:-1], sorted(list(break_points))[-2::-1]
        )
    ]

    # find live demes in each epoch, starting with most ancient
    demes_present = defaultdict(list)
    # add demes as they appear from past to present to end of lists
    deme_start_times = defaultdict(list)
    for deme in g.demes:
        deme_start_times[deme.start_time].append(deme.name)

    if math.inf not in deme_start_times.keys():
        raise ValueError("Root deme must have start time as inf")
    if len(deme_start_times[math.inf]) != 1:
        raise ValueError("Deme graph can only have a single root")

    for start_time in sorted(deme_start_times.keys())[::-1]:
        for deme_id in deme_start_times[start_time]:
            end_time = g[deme_id].end_time
            for interval in integration_times:
                if start_time >= interval[0] and end_time <= interval[1]:
                    demes_present[interval].append(deme_id)

    # dictionary of demographic events (pulses, splits, branches, mergers, and
    # admixtures) it's possible that the order of these events will matter
    # also noting here that there can be ambiguity about order of events, that will
    # change the demography... but there should always be a way to write the demography
    # in an unambiguous manner, using different verbs (e.g., two pulse events at the
    # same time with same dest can be converted to an admixture event, and split the
    # dest deme into two demes)
    demo_events = defaultdict(list)
    for pulse in demes_demo_events["pulses"]:
        event = ("pulse", pulse.source, pulse.dest, pulse.proportion)
        demo_events[pulse.time].append(event)
    for branch in demes_demo_events["branches"]:
        event = ("branch", branch.parent, branch.child)
        demo_events[branch.time].append(event)
    for merge in demes_demo_events["mergers"]:
        event = ("merge", merge.parents, merge.proportions, merge.child)
        demo_events[merge.time].append(event)
    for admix in demes_demo_events["admixtures"]:
        event = ("admix", admix.parents, admix.proportions, admix.child)
        demo_events[admix.time].append(event)
    for split in demes_demo_events["splits"]:
        event = ("split", split.parent, split.children)
        demo_events[split.time].append(event)

    # if there are any unsampled demes that end before present and do not have
    # any descendent demes, we need to add marginalization events.
    for deme_id, succs in g.successors().items():
        if deme_id not in sampled_demes and (
            len(succs) == 0
            or np.all([g[succ].start_time > g[deme_id].end_time for succ in succs])
        ):
            event = ("marginalize", deme_id)
            demo_events[g[deme_id].end_time].append(event)

    return demo_events, demes_present


def _get_root_Ne(g):
    # get root population and set Ne to root size
    for deme_id, preds in g.predecessors().items():
        if len(preds) == 0:
            root_deme = deme_id
            break
    Ne = g[root_deme].epochs[0].start_size
    return Ne


def _get_integration_parameters(g, demes_present, frozen_list, Ne=None):
    """
    Returns a list of size functions, migration matrices, integration times,
    and lists frozen demes.
    """
    nu_funcs = []
    integration_times = []
    migration_matrices = []
    frozen_demes = []

    if Ne is None:
        Ne = _get_root_Ne(g)

    for interval, live_demes in sorted(demes_present.items())[::-1]:
        # get intergration time for interval
        T = (interval[0] - interval[1]) / 2 / Ne
        if T == math.inf:
            T = 0
        integration_times.append(T)
        # get frozen attributes
        freeze = [d in frozen_list for d in live_demes]
        frozen_demes.append(freeze)
        # get nu_function or list of sizes (if all constant)
        sizes = []
        for d in live_demes:
            sizes.append(_sizes_at_time(g, d, interval))
        nu_func = _make_nu_func(sizes, T, Ne)
        nu_funcs.append(nu_func)
        # get migration matrix for interval
        mig_mat = np.zeros((len(live_demes), len(live_demes)))
        for ii, d_from in enumerate(live_demes):
            for jj, d_to in enumerate(live_demes):
                if d_from != d_to:
                    m = _migration_rate_in_interval(g, d_from, d_to, interval)
                    mig_mat[jj, ii] = 2 * Ne * m
        migration_matrices.append(mig_mat)

    return nu_funcs, migration_matrices, integration_times, frozen_demes


def _make_nu_func(sizes, T, Ne):
    """
    Given the sizes at start and end of time interval, and the size function for
    each deme, along with the integration time and reference Ne, return the
    size function that gets passed to the dadi integration routines.
    """
    if np.all([s[-1] == "constant" for s in sizes]):
        # all constant
        nu_func = [s[0] / Ne for s in sizes]
        # print(sizes,T,Ne)
        #print([ele/Ne for ele in s[:-1]],s[-1],T,Ne)
    else:
        nu_func = []
        for s in sizes:
            if s[-1] == "constant":
                assert s[0] == s[1]
                nu_func.append(lambda t, N0=s[0]: N0 / Ne)
            elif s[-1] == "linear":
                nu_func.append(
                    lambda t, N0=s[0], NF=s[1]: N0 / Ne + t / T * (NF - N0) / Ne
                )
            elif s[-1] == "exponential":
                nu_func.append(
                    lambda t, N0=s[0], NF=s[1]: (N0 / Ne) * (NF / N0) ** (t / T)
                )
            else:
                raise ValueError(f"{s[-1]} not a valid size function")
            # print([ele/Ne for ele in s[:-1]],s[-1],T,Ne)
    return nu_func


def _sizes_at_time(g, deme_id, time_interval):
    """
    Returns the start size, end size, and size function for given deme over the
    given time interval.
    """
    for epoch in g[deme_id].epochs:
        if epoch.start_time >= time_interval[0] and epoch.end_time <= time_interval[1]:
            break
    if epoch.size_function not in ["constant", "exponential", "linear"]:
        raise ValueError(
            "Can only intergrate constant, exponential, or linear size functions"
        )
    size_function = epoch.size_function

    if size_function == "constant":
        start_size = end_size = epoch.start_size

    if epoch.start_time == time_interval[0]:
        start_size = epoch.start_size
    else:
        if size_function == "exponential":
            start_size = epoch.start_size * np.exp(
                np.log(epoch.end_size / epoch.start_size)
                * (epoch.start_time - time_interval[0])
                / epoch.time_span
            )
        elif size_function == "linear":
            frac = (epoch.start_time - time_interval[0]) / epoch.time_span
            start_size = epoch.start_size + frac * (epoch.end_size - epoch.start_size)

    if epoch.end_time == time_interval[1]:
        end_size = epoch.end_size
    else:
        if size_function == "exponential":
            end_size = epoch.start_size * np.exp(
                np.log(epoch.end_size / epoch.start_size)
                * (epoch.start_time - time_interval[1])
                / epoch.time_span
            )
        elif size_function == "linear":
            frac = (epoch.start_time - time_interval[1]) / epoch.time_span
            end_size = epoch.start_size + frac * (epoch.end_size - epoch.start_size)

    # print(epoch, start_size, end_size, size_function)
    return start_size, end_size, size_function


def _migration_rate_in_interval(g, source, dest, time_interval):
    """
    Get the migration rate from source to dest over the given time interval.
    """
    rate = 0
    for mig in g.migrations:
        try:  # if asymmetric migration
            if mig.source == source and mig.dest == dest:
                if (
                    mig.start_time >= time_interval[0]
                    and mig.end_time <= time_interval[1]
                ):
                    rate = mig.rate
        except AttributeError:  # symmetric migration
            if source in mig.demes and dest in mig.demes:
                if (
                    mig.start_time >= time_interval[0]
                    and mig.end_time <= time_interval[1]
                ):
                    rate = mig.rate
    return rate


##
## Functions for SFS computation
##



def _compute_sfs(
    demo_events,
    demes_present,
    sample_sizes,
    nu_funcs,
    migration_matrices,
    integration_times,
    frozen_demes,
    pts,
    theta=1.0,
    gamma=None,
    h=None,
):
    """
    Integrates using dadi to find the SFS for given demo events, etc
    """
    if gamma is not None and h is None:
        h = 0.5

    # theta is a scalar
    assert type(theta) in [int, float]

    integration_intervals = sorted(list(demes_present.keys()))[::-1]

    # set up initial steady-state 1D phi for ancestral deme
    if gamma is None:
        gamma = 0.0
    if h is None:
        h = 0.5
    xx = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D(xx, theta0=theta, gamma=gamma, h=h)
    
    # for each set of demographic events and integration epochs, step through
    # integration, apply events, and then reorder populations to align with demes
    # present in the next integration epoch
    pop_ids = []
    i=0
    for (T, nu, M, frozen, interval) in zip(
        integration_times,
        nu_funcs,
        migration_matrices,
        frozen_demes,
        integration_intervals,
    ):
        # print('T: ',T,'\nnu: ',nu,'\nM:\n',M, '\ninterval',interval,'\nevents:',demo_events[interval[1]],'\n')
        if pop_ids == []:
            pop_ids = demes_present[interval]
        if T > 0:
            gamma_int = [gamma for _ in frozen]
            h_int = [h for _ in frozen]
            integration_params = [nu, T, M, gamma_int, h_int, theta, frozen]
            phi = _integrate_phi(phi, xx, integration_params, pop_ids)
            # ###
            # try:
            #     print(integration_params)
            # except:
            #     pass
            # ###
        events = demo_events[interval[1]]
        for event in events:
            phi, pop_ids = _apply_event(phi, xx, pop_ids, event, interval[1], sample_sizes, demes_present)

        if interval[1] > 0:
            # rearrange to next order of demes
            next_interval = integration_intervals[
                [x[0] for x in integration_intervals].index(interval[1])
            ]
            next_deme_order = demes_present[next_interval]
            # ###
            # print(pop_ids)
            # ###
            if pop_ids != next_deme_order:
                new_order = [pop_ids.index(pop)+1 for pop in next_deme_order]
                phi = dadi.PhiManip.reorder_pops(phi, new_order)
                pop_ids = next_deme_order
            # print('new order:',pop_ids,'\n\n')

    fs = dadi.Spectrum.from_phi(phi, sample_sizes, [xx]*len(pop_ids), pop_ids=pop_ids)
    return fs


def _apply_event(phi, xx, pop_ids, event, interval, sample_sizes, demes_present):
    e = event[0]
    if e == "marginalize":
        marginalize_i = pop_ids.index(event[1])
        phi = dadi.PhiManip.remove_pop(phi, xx, marginalize_i+1)
        pop_ids.pop(marginalize_i)
    elif e == "split":
        children = event[2]
        parent = event[1]
        parent_i = pop_ids.index(parent)
        if len(children) == 1:
            # "split" into just one population (name change)
            pop_ids = pop_ids[:parent_i] +children+ pop_ids[parent_i+1:]
        else:
            # split into multiple children demes
            if len(children) + len(pop_ids) - 1 > 5:
                raise ValueError("Cannot apply split that creates more than 5 demes")
            phi= _split_phi(phi, xx, pop_ids, parent)
            # When dadi splits a population, one of the new children is always the last in the phi matrix
            pop_ids = pop_ids[:parent_i] + [children[0]] + pop_ids[parent_i+1:] + [children[1]]
    elif e == "branch":
        # branch is a split, but keep the pop_id of parent
        parent = event[1]
        parent_i = pop_ids.index(parent)
        child = event[2]
        children = [parent, child]
        phi = _split_phi(phi, xx, pop_ids, parent)
        # When dadi splits a population, one of the new children is always the last in the phi matrix
        pop_ids = pop_ids[:parent_i] + [children[0]] + pop_ids[parent_i+1:] + [children[1]]
    elif e in ["admix", "merge"]:
        # two or more populations merge, based on given proportion(s)
        parents = event[1]
        proportions = event[2]
        child = event[3]
        if child not in pop_ids:
            pop_ids.append(child)
            if len(pop_ids)>5:
                raise ValueError("Cannot apply admix that creates more than 5 demes")
            phi = _admix_new_pop_phi(phi, xx, proportions, pop_ids[:-1], parents)
        else:
            # admixture from one or more populations to another existing population
            # with some proportion
            phi = _admix_phi(phi, xx, proportions, pop_ids, sources, dest)
        if e == "merge":
            for parent in parents:
                remove_i = pop_ids.index(parent)
                pop_ids.pop(remove_i)
                phi = dadi.PhiManip.remove_pop(phi, xx, remove_i+1)
    elif e == "pulse":
        # admixture from one population to another, with some proportion
        source = event[1]
        dest = event[2]
        proportion = event[3]
        phi = _admix_phi(phi, xx, proportion, pop_ids, source, dest)
    else:
        raise ValueError(f"Haven't implemented methods for event type {e}")
    # print(pop_ids)
    return phi, pop_ids

def _integrate_phi(phi, xx, integration_params, pop_ids):
    """
    Intrgates phi into children with split_sizes, from the deme at split_idx.
    """
    nu, T, M, gamma, h, theta, frozen = integration_params
    if len(pop_ids) == 1:
        phi = dadi.Integration.one_pop(
            phi, xx, T, nu[0], 
            gamma=gamma[0], h=h[0], theta0=theta, frozen=frozen[0]
            )
    elif len(pop_ids) == 2:
        phi = dadi.Integration.two_pops(
            phi, xx, T, nu1=nu[0], nu2=nu[1], m12=M[0,1], m21=M[1,0], 
            gamma1=gamma[0], gamma2=gamma[1], h1=h[0], h2=h[1], theta0=theta, 
            # initial_t=0, 
            frozen1=frozen[0], frozen2=frozen[1]
            )
    elif len(pop_ids) == 3:
        phi = dadi.Integration.three_pops(
           phi, xx, T, nu1=nu[0], nu2=nu[1], nu3=nu[2],
           m12=M[0,1], m13=M[0,2], m21=M[1,0], m23=M[1,2], m31=M[2,0], m32=M[2,1],
           gamma1=gamma[0], gamma2=gamma[1], gamma3=gamma[2], h1=h[0], h2=h[1], h3=h[2],
           theta0=theta, 
           # initial_t=0, 
           frozen1=frozen[0], frozen2=frozen[1], frozen3=frozen[2]
           )
    elif len(pop_ids) == 4:
        phi = dadi.Integration.four_pops(
            phi, xx, T, nu1=nu[0], nu2=nu[1], nu3=nu[2], nu4=nu[3],
           m12=M[0,1], m13=M[0,2], m14=M[0,3], m21=M[1,0], m23=M[1,2], m24=M[1,3], 
           m31=M[2,0], m32=M[2,1], m34=M[2,3], m41=M[3,0], m42=M[3,1], m43=M[3,2],
           gamma1=gamma[0], gamma2=gamma[1], gamma3=gamma[2], gamma4=gamma[3], h1=h[0], h2=h[1], h3=h[2], h4=h[3],
           theta0=theta, initial_t=0, frozen1=frozen[0], frozen2=frozen[1], frozen3=frozen[2], frozen4=frozen[3]
           )
    elif len(pop_ids) == 5:
        phi = dadi.Integration.five_pops(
           phi, xx, T, nu1=nu[0], nu2=nu[1], nu3=nu[2], nu4=nu[3], nu5=nu[4],
           m12=M[0,1], m13=M[0,2], m14=M[0,3], m15=M[0,4], 
           m21=M[1,0], m23=M[1,2], m24=M[1,3], m25=M[1,4],
           m31=M[2,0], m32=M[2,1], m34=M[2,3], m35=M[2,4], 
           m41=M[3,0], m42=M[3,1], m43=M[3,2], m45=M[3,4],
           m51=M[4,0], m52=M[4,1], m53=M[4,2], m54=M[4,3],
           gamma1=gamma[0], gamma2=gamma[1], gamma3=gamma[2], 
           gamma4=gamma[3], gamma5=gamma[4],
           h1=h[0], h2=h[1], h3=h[2], h4=h[3], h5=h[4],
           theta0=theta, initial_t=0, 
           frozen1=frozen[0], frozen2=frozen[1], frozen3=frozen[2], 
           frozen4=frozen[3], frozen5=frozen[3]
           )
    return phi

def _split_phi(phi, xx, pop_ids, parent):
    """
    Split the phi into children from the deme at pop_ids.index(parent).
    """
    parent_i = pop_ids.index(parent)
    if len(pop_ids) == 1:
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    elif len(pop_ids) == 2:
        phimanip_func = [dadi.PhiManip.phi_2D_to_3D_split_1, dadi.PhiManip.phi_2D_to_3D_split_2][parent_i]
        phi = phimanip_func(xx, phi)
    elif len(pop_ids) == 3:
        proportions = [[1,0,0], [0,1,0], [0,0,1]][parent_i]
        phi = dadi.PhiManip.phi_3D_to_4D(phi, proportions[0],proportions[1], xx,xx,xx,xx)
    elif len(pop_ids) == 4:
        proportions = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]][parent_i]
        phi = dadi.PhiManip.phi_4D_to_5D(phi, proportions[0],proportions[1],proportions[2], xx,xx,xx,xx,xx)
    return phi

def _admix_new_pop_phi(phi, xx, proportions, pop_ids, parents):
    """
    This function is for when admixture and mergining events result in a new population.
    Merge events remove the parental demes, while admixture events do not.
    """
    parent_i = []
    for parent in parents:
        parent_i.append(pop_ids.index(parent))
    proportion_l = _make_sorted_proportions_list(proportions, parent_i, None, pop_ids)
    if len(pop_ids) == 2:
        phi = dadi.PhiManip.phi_2D_to_3D_admix(phi, proportion_l[0], xx,xx,xx)
    if len(pop_ids) == 3:
        phi = dadi.PhiManip.phi_3D_to_4D(phi, proportion_l[0],proportion_l[1], xx,xx,xx,xx)
    if len(pop_ids) == 4:
        phi = dadi.PhiManip.phi_4D_to_5D(phi, proportion_l[0],proportion_l[1],proportion_l[2], xx,xx,xx,xx,xx)
    return phi

def _admix_phi(phi, xx, proportions, pop_ids, sources, dest):
    # Get index of source and destination populations
    # uses admix in place
    if type(proportions) != list:
        proportions = [proportions]
    if type(sources) != list:
        sources = [sources]
    source_i = []
    for source in sources:
        source_i.append(pop_ids.index(source))
    dest_i = pop_ids.index(dest)
    proportion_l = _make_sorted_proportions_list(proportions, source_i, dest_i, pop_ids)
    if len(pop_ids) == 2:
        pulse = [
        dadi.PhiManip.phi_2D_admix_2_into_1,
        dadi.PhiManip.phi_2D_admix_1_into_2
        ][dest_i]
        pulse(phi, proportions[0], xx,xx)
    if len(pop_ids) == 3:
        proportion1,proportion2 = proportion_l
        pulse = [
        dadi.PhiManip.phi_3D_admix_2_and_3_into_1,
        dadi.PhiManip.phi_3D_admix_1_and_3_into_2,
        dadi.PhiManip.phi_3D_admix_1_and_2_into_3
        ][dest_i]
        pulse(phi, proportion1,proportion2, xx,xx,xx)
    if len(pop_ids) == 4:
        proportion1,proportion2,proportion3 = proportion_l
        pulse = [
        dadi.PhiManip.phi_4D_admix_into_1,
        dadi.PhiManip.phi_4D_admix_into_2,
        dadi.PhiManip.phi_4D_admix_into_3,
        dadi.PhiManip.phi_4D_admix_into_4
        ][dest_i]
        pulse(phi, proportion1,proportion2,proportion3, xx,xx,xx,xx)
    if len(pop_ids) == 5:
        proportion1,proportion2,proportion3,proportion4 = proportion_l
        pulse = [
        dadi.PhiManip.phi_5D_admix_into_1,
        dadi.PhiManip.phi_5D_admix_into_2,
        dadi.PhiManip.phi_5D_admix_into_3,
        dadi.PhiManip.phi_5D_admix_into_4,
        dadi.PhiManip.phi_5D_admix_into_5
        ][dest_i]
        pulse(phi, proportion1,proportion2,proportion3,proportion4, xx,xx,xx,xx,xx)
    return phi

def _make_sorted_proportions_list(proportions, source_i, dest_i, pop_ids):
    proportion_l = [0] * len(pop_ids)
    for i,prop in zip(source_i,proportions):
        proportion_l[i] = prop
    try:
        proportion_l.pop(dest_i)
    except:
        pass
    # print(proportion_l)
    return proportion_l

