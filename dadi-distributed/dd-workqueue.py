#!/usr/bin/env python
import argparse
import os
import stat
import subprocess
from work_queue import *
import sys

def add_task(args, batch_num, task_num, q):
    cmd = "./infer_dm.py --syn-fs infile --model " + args.model
    if args.misid:
        cmd += " --misid"
    cmd += " --p0 " + args.p0 + " --ubounds " + args.ubounds + " --lbounds " + args.lbounds
    t = Task(cmd)
    t.specify_input_file("infer_dm.py")
    t.specify_input_file(args.infile, "infile", cache=False)
    if args.p0.startswith("output"):
        t.specify_input_file(os.path.join(args.dir, args.p0), cache=False)
    t.specify_output_file(os.path.join(args.dir, "optimization" + str(batch_num), "output.run" + str(task_num)), "output.run0")
    q.submit(t)

def add_tasks(jobs, batch_num, args, q):
    os.makedirs(os.path.join(args.dir, "optimization" + str(batch_num)), exist_ok=True)
    for i in range(jobs):
        add_task(args, batch_num, i, q)

def best_fit(args, batch_num):
    cmd = "dadi-cli BestFit --dir " + args.dir + "/optimization" + str(batch_num) + \
        " --output " + args.dir + "/output" + str(batch_num) + ".params --ubounds " + args.ubounds + " --lbounds " + args.lbounds
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if out.returncode != 0:
        print("BestFit returned an error")
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

    os.makedirs(args.dir, exist_ok=True)
    try:
        q = WorkQueue(port = WORK_QUEUE_DEFAULT_PORT) #, debug_log = "debug.log")
    except:
        print("Instantiation of Work Queue failed!")
        sys.exit(1)
    print("listening on port %d..." % q.port)
    jobs = 1
    if args.jobs:
        jobs = args.jobs
    batch_num = 1
    while True:
        add_tasks(jobs, batch_num, args, q)
        print("waiting for tasks to complete...")
        while not q.empty():
            t = q.wait(5)
            if t:
                if t.return_status != 0:
                    print("problem encountered, return status =" + str(t.return_status))
                    sys.exit(t.return_status)
                o = best_fit(args, batch_num)
                print(o)
                if o.find("CONVERGED RESULT FOUND!") != -1:
                    q.shutdown_workers(0)
                    sys.exit(0)
                args.p0 = "output" + str(batch_num) + ".params"
