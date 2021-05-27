#!/usr/bin/env python
import argparse
import os
import stat
import subprocess
from work_queue import *
import sys

def add_task(args, task_num, q):
    print("adding task for p0=" + str(args.p0))
    script = "./inferdm" + str(task_num) + ".sh"
    t = Task(script)
    t.specify_output_file("std.out")
    t.specify_output_file("std.err")
    if args.p0.startswith("output"):
        t.specify_input_file(os.path.join(args.dir, args.p0), cache=False)
    t.specify_tag("dadi")
    script = os.path.join(args.dir, script)
    write_script(script, args)
    t.specify_input_file(script, cache=False)
    t.specify_input_file(args.infile, "infile", cache=False)
    t.specify_output_file(os.path.join(args.dir, "outfiles.tar.gz"), cache=False)
    taskid = q.submit(t)
    print("submitted task (id# %d): %s" % (taskid, t.command))

def best_fit(args, task_num):
    cmd = "dadi-cli BestFit --dir " + args.dir + "/optimization" + str(task_num) + \
        " --output " + args.dir + "/output" + str(task_num) + ".params --ubounds " + args.ubounds + " --lbounds " + args.lbounds
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if out.returncode != 0:
        print("BestFit returned an error")
        print(out.stderr)
        sys.exit(1)
    return out.stdout

def unpack(args, task_num):
    d = os.getcwd()
    os.chdir(args.dir)
    o = "optimization" + str(task_num)
    os.makedirs(o, exist_ok=True)
    out = subprocess.run("tar xzf outfiles.tar.gz -C " + o, shell=True)
    if out.returncode != 0:
        sys.exit(1)
    os.remove("outfiles.tar.gz")
    os.chdir(d)

def write_script(script, args):
    f = open(script, "w")
    f.write("dadi-cli InferDM --syn-fs infile --model " + args.model)
    if args.misid:
        f.write(" --misid")
    f.write(" --p0 " + args.p0 + " --ubounds " + args.ubounds + " --lbounds " + args.lbounds + " --output outfile")
    if args.jobs:
        f.write(" --jobs " + str(args.jobs))
    f.write(" >std.out 2>std.err")
    f.write("\ntar czf outfiles.tar.gz outfile.run*")
    f.close()
    os.chmod(script, stat.S_IRWXU)

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
        q = WorkQueue(port = WORK_QUEUE_DEFAULT_PORT) #, debug_log = "debug.log")
    except:
        print("Instantiation of Work Queue failed!")
        sys.exit(1)
    print("listening on port %d..." % q.port)

    task_num = 1
    add_task(args, task_num, q)
    print("waiting for tasks to complete...")
    while not q.empty():
        t = q.wait(5)
        if t:
            if t.return_status != 0:
                print("problem encountered, return status =" + str(t.return_status))
                sys.exit(t.return_status)
            unpack(args, t.id)
            o = best_fit(args, t.id)
            print(o)
            if o.find("CONVERGED RESULT FOUND!") != -1:
                q.cancel_by_tasktag("dadi")
                q.shutdown_workers(0)
                sys.exit(0)
            else:
                args.p0 = "output" + str(t.id) + ".params"
                task_num += 1
                add_task(args, task_num, q)
    print("all tasks complete!")
    sys.exit(0)

