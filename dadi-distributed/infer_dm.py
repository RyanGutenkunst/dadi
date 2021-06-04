#!/usr/bin/env python
import argparse
       
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Determine whether using GPUs to accelerate inference or not; Default: False')
    parser.add_argument('--constants', type=float, nargs='+', help='The fixed parameters during the inference, please use -1 to indicate a parameter is NOT fixed; Default: None')
    parser.add_argument('--syn-fs', type=str, required=True, help='The frequency spectrum of synonymous mutations used for inference; To generate the frequency spectrum, please use `dadi-cli GenerateFs`', dest='syn_fs')
    parser.add_argument('--grids', type=int, nargs=3, help='The sizes of the grids; Default: [sample_size[0]+10, sample_size[0]+20, sample_size[0]+30]')
    parser.add_argument('--lbounds', type=float, nargs='+', required=True, help='The lower bounds of the inferred parameters, please use -1 to indicate a parameter without lower bound')
    parser.add_argument('--misid', default=False, action='store_true', help='Determine whether adding a parameter for misidentifying ancestral alleles or not; Default: False')
    parser.add_argument('--model', type=str, required=True, help='The name of the demographic model; To check available demographic models, please use `dadi-cli Model`')
    parser.add_argument('--p0', type=str, nargs='+', required=True, help='The initial parameters for inference')
    parser.add_argument('--ubounds', type=float, nargs='+', required=True, help='The upper bounds of the inferred parameters, please use -1 to indicate a parameter without upper bound')

    args = parser.parse_args()

    def check_params(params):
        new_params = []
        for p in params:
            if p == -1.0: new_params.append(None)
            else: new_params.append(p)
        return new_params

    def read_params(params):
        new_params = []
        line = open(params, 'r').readline().rstrip().split()
        for p in line:
            new_params.append(float(p))
        return new_params[1:-1]

    def parse_params(params):
        new_params = []
        for p in params:
            new_params.append(float(p))
        return new_params

    if args.constants != None: args.constants = check_params(args.constants)
    if args.lbounds != None: args.lbounds = check_params(args.lbounds)
    if args.ubounds != None: args.ubounds = check_params(args.ubounds)

    if len(args.p0) == 1: args.p0 = read_params(args.p0[0])
    else: args.p0 = parse_params(args.p0)

    from src.InferDM import infer_demography

    infer_demography(args.syn_fs, args.model, args.grids, args.p0, "output", args.ubounds, args.lbounds, args.constants, args.misid, args.cuda)

if __name__ == '__main__':
    main()
