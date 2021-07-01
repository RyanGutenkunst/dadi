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
    #cmd += " >std.out 2>std.err"
    t = Task(cmd)
    #t.specify_output_file("std.out")
    #t.specify_output_file("std.err")
    t.specify_input_file(os.path.dirname(__file__) + "/infer_dm.py")
    t.specify_input_file(args.infile, "infile", cache=False)
    if args.p0.startswith("output"):
        t.specify_input_file(os.path.join(args.dir, args.p0), cache=False)
    local_file = os.path.join(args.dir, "optimization" + str(batch_num), "output.run" + str(task_num))
    t.specify_output_file(local_file, remote_name="output")
    q.submit(t)

def add_tasks(jobs, batch_num, args, q):
    os.makedirs(os.path.join(args.dir, "optimization" + str(batch_num)), exist_ok=True)
    for i in range(jobs):
        add_task(args, batch_num, i, q)

def best_fit(args, batch_num, q):
    cmd = "dadi-cli BestFit --dir " + args.dir + "/optimization" + str(batch_num)
    cmd += " --output " + args.dir + "/output" + str(batch_num) + ".params"
    cmd += " --ubounds " + args.ubounds + " --lbounds " + args.lbounds
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if out.returncode != 0:
        print("BestFit returned an error")
        print(out.stderr)
        sys.exit(1)
    if out.stdout.find("CONVERGED RESULT FOUND!") != -1:
        print(out.stdout)
        sys.exit(0)

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

    if not os.path.isfile(args.infile):
        print(args.infile + " not found")
        sys.exit(1)
    os.makedirs(args.dir, exist_ok=True)
    try:
        q = WorkQueue(name = "dadi-distributed")
        #q = WorkQueue(name = "dadi-distributed", debug_log = "debug.log")
        q.specify_password_file('pwfile')
    except:
        print("Instantiation of Work Queue failed!")
        sys.exit(1)
    #print("listening on port %d..." % q.port)
    jobs = 1
    if args.jobs:
        jobs = args.jobs
    batch_num = 1
    while True:
        add_tasks(jobs, batch_num, args, q)
        print("waiting for tasks to complete for batch " + str(batch_num) + "...")
        while not q.empty():
            t = q.wait(5)
            if t:
                if t.return_status != 0:
                    print("problem encountered, return status =" + str(t.return_status))
                    sys.exit(t.return_status)
        best_fit(args, batch_num, q)
        args.p0 = "output" + str(batch_num) + ".params"
        batch_num += 1
