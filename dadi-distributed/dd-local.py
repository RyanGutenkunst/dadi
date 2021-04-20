#!/usr/bin/env python
import argparse
import os
import subprocess
import sys

def best_fit(args, task_num):
    cmd = "dadi-cli BestFit --dir " + args.dir + "/optimization" + str(task_num) + \
        " --output " + args.dir + "/output" + str(task_num) + ".params --ubounds " + args.ubounds + " --lbounds " + args.lbounds
    print(cmd)
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if out.returncode != 0:
        print("BestFit returned an error")
        print(out.stderr)
        sys.exit(1)
    return out.stdout

def infer_dm(args, task_num):
    os.makedirs(args.dir + "/optimization" + str(task_num), exist_ok=True)
    cmd = "dadi-cli InferDM --syn-fs " + args.infile + " --model " + args.model
    if args.misid:
        cmd += " --misid"
    cmd += " --p0 " + args.p0 + " --ubounds " + args.ubounds + \
        " --lbounds " + args.lbounds + " --output " + args.dir + "/optimization" + str(task_num) + "/output"
    if args.jobs:
        cmd += " --jobs " + str(args.jobs)
    print(cmd)
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if out.returncode != 0:
        print("InferDM returned an error")
        print(out.stderr)
        sys.exit(1)
    return out.stdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work Queue manager for Dadi.')
    parser.add_argument('dir', help='directory in which optimization and params files are stored')
    parser.add_argument('infile', help='allele frequency spectrum file')
    parser.add_argument('--model', required=True)
    parser.add_argument('--misid', action='store_true')
    parser.add_argument('--p0', required=True)
    parser.add_argument('--ubounds', required=True)
    parser.add_argument('--lbounds', required=True)
    parser.add_argument('--jobs', type=int)
    args = parser.parse_args()

    print("running...")
    os.makedirs(args.dir, exist_ok=True)
    task_num = 1
    while True:
        o = infer_dm(args, task_num)
        print(o)
        o = best_fit(args, task_num)
        print(o)
        if o.find("CONVERGED RESULT FOUND!") != -1:
            sys.exit(0)
        else:
            args.p0 = "output" + str(task_num) + ".params"
            task_num += 1
    sys.exit(0)
