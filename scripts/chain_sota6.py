import subprocess, time, os, datetime

PY   = "/root/miniconda3/bin/python"
REPO = "/root/autodl-tmp/DeepfakeBench"
CFG  = "/root/autodl-tmp/stat_cfgs/sota_%s_s1024.yaml"
LOG  = "/root/autodl-tmp/sota_%s_s1024.log"
ST   = "/root/autodl-tmp/chain_sota6.status"
OUT  = "/root/autodl-tmp/sota_final.txt"
ALL  = ["xception","core","f3net","spsl","srm","capsule_net","sia","ffd","iid","rfm"]
GROUPS = [["srm","capsule_net"], ["sia","ffd"], ["iid","rfm"]]

def log(m):
    with open(ST,"a") as f:
        f.write("[%s] %s\n" % (datetime.datetime.now().strftime("%m-%d %H:%M"), m))

def running(n):
    r = subprocess.run(["pgrep","-f","sota_%s_s1024.yaml"%n], capture_output=True, text=True)
    return bool(r.stdout.split())

def wait_start(names, tmin):
    t0 = time.time()
    while time.time()-t0 < tmin*60:
        if all(running(n) for n in names): return True
        time.sleep(30)
    return False

def wait_end_names(names):
    while any(running(n) for n in names):
        time.sleep(60)

def remain_min(target="05:45"):
    now = datetime.datetime.now()
    h, m = [int(x) for x in target.split(":")]
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if t <= now: t += datetime.timedelta(days=1)
    return int((t-now).total_seconds()//60)

def launch(names):
    env = dict(os.environ); env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    ps = {}
    for n in names:
        ps[n] = subprocess.Popen([PY,"training/train.py","--detector_path",CFG%n],
            stdout=open(LOG%n,"w"), stderr=subprocess.STDOUT,
            cwd=REPO, start_new_session=True, env=env)
    return ps

def wait_ps(ps):
    for p in ps.values(): p.wait()

def dump():
    lines = []
    for n in ALL:
        p = LOG % n
        if not os.path.exists(p):
            lines.append("### %s : LOG_MISSING" % n); continue
        raw = open(p, errors="replace").read().replace("\r","\n")
        hits = [l for l in raw.split("\n") if "dataset: avg" in l and "step:" in l]
        lines.append("### %s : %d points" % (n, len(hits)))
        lines.extend(hits); lines.append("")
    open(OUT,"w").write("\n".join(lines))
    open(OUT+".done","w").write("done")

log("chain6 start (2-up only, 3-up abandoned after srm experiment)")
if wait_start(["f3net","spsl"], 240): log("batch2 f3net+spsl up")
else: log("WARN batch2 never started - chain2 dead?")
wait_end_names(["f3net","spsl"])
wait_end_names(["xception","core"])
log("batch2 done, batch1 confirmed gone")

for g in GROUPS:
    if remain_min() >= 130:
        log("launch %s remain=%dmin" % (",".join(g), remain_min()))
        ps = launch(g); wait_ps(ps)
        log("done %s" % ",".join(g))
    else:
        log("SKIP %s remain=%dmin" % (",".join(g), remain_min()))

log("QUEUE6_DONE")
dump()
log("dumped to sota_final.txt")
