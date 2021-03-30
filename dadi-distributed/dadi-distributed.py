#!/usr/bin/env python
from work_queue import *
import os
import stat
import subprocess
import sys

model = "two_epoch_1d"
initial_p0 = "1 1 .5"
ubounds = "10 10 1"
lbounds = "10e-3 10e-3 10e-5"
jobs = 28
task_num = 1

def add_task(dir, infile, p0):
    global task_num
    print("adding task for " + str(p0))
    script = "./inferdm" + str(task_num) + ".sh"
    t = Task(script)
    if p0 != initial_p0:
        p0 = "output" + str(p0) + ".params"
        t.specify_input_file(os.path.join(dir, p0), cache=False)
    t.specify_tag("dadi")
    script = os.path.join(dir, script)
    write_script(script, p0)
    t.specify_input_file(script, cache=False)
    t.specify_input_file(infile, "infile", cache=False)

    t.specify_output_file(os.path.join(dir, "outfiles.tar.gz"), cache=False)
    taskid = q.submit(t)
    print("submitted task (id# %d): %s" % (taskid, t.command))
    task_num += 1

def best_fit(dir, task_num):
    cmd = "dadi-cli BestFit --dir " + dir + "/optimization" + str(task_num) + \
        " --output " + dir + "/output" + str(task_num) + ".params --ubounds " + ubounds + " --lbounds " + lbounds
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if out.returncode != 0:
        print("BestFit returned an error")
        print(out.stderr)
        sys.exit(1)
    return out.stdout

def unpack(dir, task_num):
    d = os.getcwd()
    os.chdir(dir)
    o = "optimization" + str(task_num)
    os.makedirs(o, exist_ok=True)
    out = subprocess.run("tar xzf outfiles.tar.gz -C " + o, shell=True)
    if out.returncode != 0:
        sys.exit(1)
    os.remove("outfiles.tar.gz")
    os.chdir(d)

def write_script(script, p0):
    f = open(script, "w")
    f.write("dadi-cli InferDM --syn-fs infile --model " + model + " --misid --p0 " + p0 + " --ubounds " + ubounds + \
        " --lbounds " + lbounds + " --output outfile --jobs " + str(jobs) + "\n")
    f.write("tar czf outfiles.tar.gz outfile.run*")
    f.close()
    os.chmod(script, stat.S_IRWXU)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("dadi <work dir> <fs file>")
        sys.exit(1)

    try:
        q = WorkQueue(port = WORK_QUEUE_DEFAULT_PORT, debug_log = "debug.log")
    except:
        print("Instantiation of Work Queue failed!")
        sys.exit(1)
    print("listening on port %d..." % q.port)

    dir = sys.argv[1]
    infile = sys.argv[2]
    add_task(dir, infile, initial_p0)
    print("waiting for tasks to complete...")
    while not q.empty():
        t = q.wait(5)
        if t:
            unpack(dir, t.id)
            o = best_fit(dir, t.id)
            print(o)
            if o.find("CONVERGED RESULT FOUND!") != -1:
                q.cancel_by_tasktag("dadi")
                q.shutdown_workers(0)
                sys.exit(0)
            else:
                add_task(dir, infile, t.id)
    print("all tasks complete!")
    sys.exit(0)

