#!/usr/bin/env python
"""
Convert the output of ms to an N-dimensional frequency spectrum.
"""
import sys

import dadi

if __name__ == '__main__':
    average = ('-av' in sys.argv)
    
    input = sys.stdin
    output = sys.stdout
    sfs,header = dadi.Spectrum.from_ms_file(input, average, mask_corners=True,
                                            return_header=True)
    
    dadi.IO.sfs_to_file(sfs, output, comment_lines=header)
