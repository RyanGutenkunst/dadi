# Bootstrapping

dadi's likelihood function treats all variants as independent, but if your data are linked then the variants are not idenpendent. In that case, standard likelihood theory should not be used to estimate parameter uncertainties and significance levels for hypothesis tests. 

To perform such statistical tests, we must bootstrap over our data. To do so, we divide the data into independent chunks and generate new pseudo-replicate data sets by sampling with replacement from those chunks. For genomic data, we typically do this by dividing the genome into large (several Mb) segments, so linkage within the segments is much greater than linkage between them. To do this, we start from the data dictionary `dd` associated with our data file and then fragment it into chunks
```python
chunks = dadi.Misc.fragment_data_dict(dd, chunk_size)
```
From those chunks, we can then create boostrap spectra.
```python
boots = dadi.Misc.bootstraps_from_dd_chunks(chunks, Nboot, pop_ids, ns)
```

If subsampling the data, to avoid projection, we need to ensure each bootstrap replicate gets a different subsample. So we must do the slower process are parsing the VCF file for each bootstrap, using the method `dadi.Misc.bootstraps_subsample_vcf`.