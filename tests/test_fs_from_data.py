import pytest
import dadi
import numpy as np

@pytest.fixture
def test_details():
    pytest.datafile = 'examples/fs_from_data/1KG.YRI.CEU.biallelic.synonymous.snps.withanc.strict.subset.vcf.gz'
    pytest.popfile = 'examples/fs_from_data/1KG.YRI.CEU.popfile.txt'
    pytest.pop_ids, pytest.ns = ['YRI','CEU'], [20,24]
    pytest.chunk_size = 1e6

def test_basic_loading(test_details):
    dd = dadi.Misc.make_data_dict_vcf(pytest.datafile, pytest.popfile)
    fs = dadi.Spectrum.from_data_dict(dd, pytest.pop_ids, pytest.ns)

    # Check a couple of arbitrary entries
    assert(np.allclose(fs[1,2], 3.1208634510568456))
    assert(np.allclose(fs[6,5], 0.744764040575882))

def test_saving_loading(test_details):
    dd = dadi.Misc.make_data_dict_vcf(pytest.datafile, pytest.popfile)
    fs = dadi.Spectrum.from_data_dict(dd, pytest.pop_ids, pytest.ns)
    fs.to_file('test.fs')

    fs2 = dadi.Spectrum.from_file('test.fs')
    assert(np.allclose(fs, fs2))

def test_folded_loading(test_details):
    dd = dadi.Misc.make_data_dict_vcf(pytest.datafile, pytest.popfile)
    fs_folded = dadi.Spectrum.from_data_dict(dd, pytest.pop_ids, pytest.ns, polarized=False)

    # Check a couple of arbitrary entries
    assert(fs_folded.mask[10,16])
    assert(np.allclose(fs_folded[6,5], 0.995723483283639))

def test_chunking(test_details):
    """
    Test that chunks are correctly sized.

    Based on bug fixed on October 14, 2020.
    """
    dd = dadi.Misc.make_data_dict_vcf(pytest.datafile, pytest.popfile)
    fragments = dadi.Misc.fragment_data_dict(dd, pytest.chunk_size)
    # To find bad spacings in chunks, check that that sequential
    # chunks 1,2,3, the gap between snps in 3 and 1 is at least
    # chunk_size.
    for ii, f1 in enumerate(fragments[:-2]):
        f2, f3 = fragments[ii+1], fragments[ii+2]
        try:
            k1 = list(f1)[0]
            k3 = list(f3)[0]
        except IndexError:
            # Skip empty chunks
            continue
        # Get positions
        chr1, chr3 = k1.split('_')[0], k3.split('_')[0]
        if chr1 != chr3:
            # Ignore cases in which chunks are on different chromosomes
            continue
        pos1 = int(k1.split('_')[1])
        pos3 = int(k3.split('_')[1])
        assert(pos3 - pos1 >= pytest.chunk_size)

def test_chunking_naming(test_details):
    """
    Test that chunked data dictionaries maintain a correct naming convention from data dictionary.

    Based on bug fixed on July 27, 2022.
    """
    import pickle
    dd = pickle.load(open('tests/test_data/complex.chromosome.naming.bpkl','rb'))
    fragments = dadi.Misc.fragment_data_dict(dd, pytest.chunk_size)
    dd_keys = list(dd.keys())
    fragments_keys = []
    for ele in [list(ele_dd.keys()) for ele_dd in fragments]:
        fragments_keys.extend(ele)
    dd_keys.sort()
    fragments_keys.sort()
    for key1, key2 in zip(dd_keys, fragments_keys):
        assert(key1==key2)


def test_boostraps(test_details):
    dd = dadi.Misc.make_data_dict_vcf(pytest.datafile, pytest.popfile)
    fragments = dadi.Misc.fragment_data_dict(dd, pytest.chunk_size)
    boots = dadi.Misc.bootstraps_from_dd_chunks(fragments, 100, pytest.pop_ids, pytest.ns)

    # Test that size of bootstraps is reasonable
    meanS = np.mean([_.S() for _ in boots])
    assert(500 < meanS < 600)

def test_boostraps_folded(test_details):
    dd = dadi.Misc.make_data_dict_vcf(pytest.datafile, pytest.popfile)
    fragments = dadi.Misc.fragment_data_dict(dd, pytest.chunk_size)
    boots = dadi.Misc.bootstraps_from_dd_chunks(fragments, 1, pytest.pop_ids, pytest.ns, polarized=False)

    assert(boots[0].mask[-1,-2])

def test_subsample(test_details):
    dd_subsample = dadi.Misc.make_data_dict_vcf(pytest.datafile, pytest.popfile,
                                                subsample={'YRI': pytest.ns[0]//2, 'CEU': pytest.ns[1]//2})
    fs_subsample = dadi.Spectrum.from_data_dict(dd_subsample, pytest.pop_ids, pytest.ns)

    # Test that we haven't introduced any projection, by ensuring all non-zero
    # entries are >= 1
    assert(fs_subsample[fs_subsample != 0].min() >= 1.0)

def test_subsample_bootstrap(test_details):
    # Just test that this runs
    boots_subsample = dadi.Misc.bootstraps_subsample_vcf(pytest.datafile, pytest.popfile,
                                                         subsample={'YRI': pytest.ns[0]//2, 'CEU': pytest.ns[1]//2}, Nboot=2, 
                                                         chunk_size=pytest.chunk_size, pop_ids=pytest.pop_ids)
