#!/usr/bin/env python
"""
Convert ms output to a frequency spectrum.

Reads the ms results from stdin and outputs to stdout.

If the '-av' flag is passed, the output is the average spectrum of the ms
iterations. Otherwise it is the sum.

For example: 'ms <details> | ms_jsfs.py -av > output.fs'
"""

if __name__ == '__main__':
    import sys
    import dadi

    average = ('-av' in sys.argv)
    
    input = sys.stdin
    output = sys.stdout
    sfs,header = dadi.Spectrum.from_ms_file(input, average, mask_corners=True,
                                            return_header=True)
    
    sfs.to_file(output, comment_lines=header)
