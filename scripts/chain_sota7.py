import subprocess, time, os, datetime

PY   = "/root/miniconda3/bin/python"
REPO = "/root/autodl-tmp/DeepfakeBench"
CFG  = "/root/autodl-tmp/stat_cfgs/sota_%s_s1024.yaml"
LOG  = "/root/autodl-tmp/sota_%s_s1024.log"
ST   = "/root/autodl-tmp/chain_sota7.status"
OUT  = "/root/autodl-tmp/sota_final.txt"
ALL  = ["xception","core","f3net","spsl","srm","capsule_net","sia","ffd","iid","rfm"]
TODO = ["core","f3net","spsl","srm","capsule_net","sia","ffd"]

def log(m):
    with open(ST,"a") as f:
        f.write("[" + datetime.datetime.now().strftime("%m-%d %H:%M") + "] " + m + "\n")

def remain_min(target="05:45"):
    now = datetime.datetime.now()
    h, m = [int(x) for x in target.split(":")]
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if t <= now:
        t += datetime.timedelta(days=1)
    return int((t - now).total_seconds() // 60)

def mem_gib():
    try:
        return int(open("/sys/fs/cgroup/memory.current").read()) / 1073741824.0
    except Exception:
        return 0.0

def running(n):
    r = subprocess.run(["pgrep","-f","sota_" + n + "_s1024.yaml"], capture_output=True, text=True)
    return bool(r.stdout.split())

def run_one(n):
    log("start " + n + " remain=" + str(remain_min()) + "min mem=" + ("%.1f" % mem_gib()))
    env = dict(os.environ)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    p = subprocess.Popen([PY, "training/train.py", "--detector_path", CFG % n],
        stdout=open(LOG % n, "w"), stderr=subprocess.STDOUT,
        cwd=REPO, start_new_session=True, env=env)
    p.wait()
    log("done " + n + " rc=" + str(p.returncode) + " mem=" + ("%.1f" % mem_gib()))

def dump():
    lines = []
    for n in ALL:
        p = LOG % n
        if not os.path.exists(p):
            lines.append("### " + n + " : LOG_MISSING"); continue
        raw = open(p, errors="replace").read().replace(chr(13), chr(10))
        hits = [l for l in raw.split(chr(10)) if "dataset: avg" in l and "step:" in l]
        lines.append("### " + n + " : " + str(len(hits)) + " points")
        lines.extend(hits); lines.append("")
    open(OUT, "w").write(chr(10).join(lines))
    open(OUT + ".done", "w").write("done")

log("chain7 start: 1-up sequential (2-up ruled out, cgroup 62GiB wall)")
while running("xception"):
    time.sleep(60)
log("xception finished, mem=" + ("%.1f" % mem_gib()))
for n in TODO:
    if remain_min() >= 85:
        run_one(n)
    else:
        log("SKIP " + n + " remain=" + str(remain_min()) + "min")
log("QUEUE7_DONE")
dump()
log("dumped to " + OUT)
