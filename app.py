#!/usr/bin/env python3
"""
Render-hosted remote control for the Kaggle Applio kernel.
Serves the web UI, exposes /api/start, /api/status, /api/stop.
State lives in Redis. The browser connects to Gradio directly.
"""
import json, os, re, secrets, subprocess, tempfile, threading, time, urllib.request
from pathlib import Path
from flask import Flask, Response, jsonify, request
import redis

KAGGLE_USER = os.environ.get("KAGGLE_USER", "").strip()
KERNEL_SLUG = os.environ.get("KERNEL_SLUG", "rvc-gpu-server")
CACHE_SLUG  = os.environ.get("CACHE_SLUG",  "rvc-cache")
DATASET     = os.environ.get("RVC_DATASET", "supporttopal/sweet-female-rvc")
DEFAULT_MINUTES = int(os.environ.get("RVC_MINUTES", "60"))
NTFY = "https://ntfy.sh"
REDIS_URL = os.environ.get("REDIS_URL", "").strip()

if not KAGGLE_USER or not REDIS_URL:
    raise SystemExit("Set KAGGLE_USER and REDIS_URL env vars.")

KERNEL_ID = f"{KAGGLE_USER}/{KERNEL_SLUG}"
CACHE_ID  = f"{KAGGLE_USER}/{CACHE_SLUG}"
DONE = {"COMPLETE", "ERROR", "CANCELACKNOWLEDGED", "CANCELLED"}
BUILD = Path(tempfile.gettempdir()) / "rvc_kernel_build"

rdb = redis.from_url(REDIS_URL, decode_responses=True)
app = Flask(__name__)

# ---------- Redis helpers ----------
def rget(k, d=None):
    v = rdb.get(f"rvc:{k}")
    if v is None: return d
    try: return json.loads(v)
    except Exception: return v

def rset(k, v, ttl=None):
    rdb.set(f"rvc:{k}", json.dumps(v))
    if ttl: rdb.expire(f"rvc:{k}", ttl)

def rdel(k): rdb.delete(f"rvc:{k}")

# ---------- Kaggle / ntfy ----------
def kaggle(*args, timeout=120):
    try:
        p = subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 1, "kaggle command timed out"

def parse_state(out):
    m = re.search(r'status\s+"?([\w.]+)"?', out or "")
    return m.group(1).split(".")[-1].replace("_", "").upper() if m else None

def run_state(kernel_id=KERNEL_ID):
    code, out = kaggle("kernels", "status", kernel_id)
    return parse_state(out) if code == 0 else None

def is_active(s): return s is not None and s not in DONE

def ntfy_read(topic):
    try:
        with urllib.request.urlopen(f"{NTFY}/{topic}/json?poll=1&since=all", timeout=15) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:
        return []
    out = []
    for line in text.splitlines():
        try: m = json.loads(line)
        except ValueError: continue
        if m.get("event") == "message": out.append(m)
    return out

def ntfy_send(topic, text):
    req = urllib.request.Request(f"{NTFY}/{topic}", data=text.encode(), method="POST")
    urllib.request.urlopen(req, timeout=15).read()

def find_link(msgs):
    link = None
    for m in msgs:
        s = m.get("message", "")
        if s.startswith("LINK "): link = s[5:].strip()
    return link

def write_kernel(build_dir, kernel_id, slug, code, gpu, dataset_sources, kernel_sources):
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "script.py").write_text(code)
    meta = {
        "id": kernel_id, "title": slug.replace("-", " "), "code_file": "script.py",
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true" if gpu else "false", "enable_internet": "true",
        "dataset_sources": dataset_sources, "competition_sources": [],
        "kernel_sources": kernel_sources,
    }
    (build_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

# ---------- Kernel templates (same as rvc_gpu.py) ----------
CACHE_TEMPLATE = r'''
import os, subprocess, sys
APPLIO = "/kaggle/working/Applio"; LIBS = "/kaggle/working/pylibs"
def sh(cmd, cwd=None):
    print(">>", cmd, flush=True); subprocess.run(cmd, shell=True, check=True, cwd=cwd)
sh("pip install -q uv")
sh(f"git clone --depth 1 https://github.com/IAHispano/Applio.git {APPLIO}")
sh(f"uv pip install -q --python {sys.executable} --target {LIBS} -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match", cwd=APPLIO)
env = dict(os.environ, PYTHONPATH=LIBS)
subprocess.run([sys.executable, "-c", "import torch, gradio; print('torch', torch.__version__, 'gradio', gradio.__version__)"], check=True, env=env)
sh(f"rm -rf {APPLIO}")
print("cache ready", flush=True)
'''

KERNEL_TEMPLATE = r'''
import glob, json, os, re, shutil, subprocess, sys, threading, time, urllib.request
TOPIC = "__TOPIC__"; MAX_SECONDS = __MINUTES__ * 60; USE_CACHE = __USE_CACHE__
MODEL_NAME = "sweet_female"; APPLIO = "/kaggle/working/Applio"; NTFY = "https://ntfy.sh"
started = time.time(); ENV = dict(os.environ, PYTHONUNBUFFERED="1"); proc = None

def say(msg):
    try:
        req = urllib.request.Request(f"{NTFY}/{TOPIC}", data=msg.encode(), method="POST")
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as e: print("ntfy post failed:", e, flush=True)

def sh(cmd, cwd=None):
    print(">>", cmd, flush=True); subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=ENV)

def stop_requested():
    try:
        with urllib.request.urlopen(f"{NTFY}/{TOPIC}/json?poll=1&since=all", timeout=20) as r:
            for raw in r.read().decode("utf-8", "replace").splitlines():
                try: m = json.loads(raw)
                except ValueError: continue
                if m.get("event") == "message" and m.get("message", "").strip() == "STOP": return True
    except Exception as e: print("poll failed:", e, flush=True)
    return False

def kill_everything(reason):
    say("STATUS " + reason)
    try:
        if proc is not None: proc.terminate()
    except Exception: pass
    try:
        import psutil
        for child in psutil.Process().children(recursive=True):
            try: child.kill()
            except Exception: pass
    except Exception:
        subprocess.run(["pkill", "-P", str(os.getpid())])
    say("STATUS finished"); os._exit(0)

def watchdog():
    tick = 0
    while True:
        time.sleep(15); tick += 1
        if time.time() - started > MAX_SECONDS: kill_everything("time limit reached, shutting down")
        if stop_requested(): kill_everything("stop received, shutting down")
        if tick % 4 == 0: say(f"STATUS alive {int((time.time() - started) / 60)} min")

threading.Thread(target=watchdog, daemon=True).start()

libs = None
if USE_CACHE:
    hits = glob.glob("/kaggle/input/*/pylibs") + glob.glob("/kaggle/input/*/*/pylibs")
    if hits:
        libs = hits[0]
        ENV["PYTHONPATH"] = libs + os.pathsep + ENV.get("PYTHONPATH", "")
        print("using cached libraries at", libs, flush=True)

try:
    say("STATUS using cached libraries" if libs else "STATUS no cache found, doing full install")
    sh(f"git clone --depth 1 https://github.com/IAHispano/Applio.git {APPLIO}")
    sh("apt-get update -y"); sh("apt-get install -y portaudio19-dev libportaudio2")
    if libs is None:
        sh("pip install -q uv")
        sh("uv pip install -q -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match --system", cwd=APPLIO)
    say("STATUS downloading Applio models")
    sh(f"{sys.executable} core.py prerequisites --models", cwd=APPLIO)

    say("STATUS copying voice files")
    dest = f"{APPLIO}/logs/{MODEL_NAME}"; os.makedirs(dest, exist_ok=True)
    for ext in ("pth", "index"):
        for f in glob.glob(f"/kaggle/input/**/*.{ext}", recursive=True):
            shutil.copy(f, dest); print("copied", f, flush=True)

    say("STATUS starting Applio")
    proc = subprocess.Popen([sys.executable, "-u", "app.py", "--listen", "--share", "--client"],
        cwd=APPLIO, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    found = {"url": None}
    def reader():
        for line in proc.stdout:
            print(line, end="", flush=True)
            m = re.search(r"https://[a-z0-9-]+\.gradio\.live", line)
            if m and not found["url"]:
                found["url"] = m.group(0); say("LINK " + found["url"])
    threading.Thread(target=reader, daemon=True).start()
    while proc.poll() is None: time.sleep(5)
    say("STATUS Applio exited on its own")
except Exception as e:
    say(f"STATUS failed: {e}"); raise
finally:
    if proc is not None and proc.poll() is None: proc.terminate()
    say("STATUS finished")
'''

# ---------- Routes ----------
@app.get("/")
def index(): return Response(PAGE, mimetype="text/html")

@app.post("/api/start")
def api_start():
    body = request.get_json(silent=True) or {}
    cpu = bool(body.get("cpu", False))
    minutes = int(body.get("minutes", DEFAULT_MINUTES))
    no_cache = bool(body.get("no_cache", False))

    s = run_state()
    if is_active(s):
        return jsonify(ok=False, error=f"A run is already active on Kaggle ({s}).", state=s)

    topic = rget("topic") or ("rvc" + secrets.token_hex(12))
    use_cache = (not no_cache) and bool(rget("cache_ready", False))

    code = (KERNEL_TEMPLATE.replace("__TOPIC__", topic)
            .replace("__MINUTES__", str(minutes))
            .replace("__USE_CACHE__", "True" if use_cache else "False"))
    write_kernel(BUILD, KERNEL_ID, KERNEL_SLUG, code, not cpu,
                 [DATASET], [CACHE_ID] if use_cache else [])

    rc, out = kaggle("kernels", "push", "-p", str(BUILD), "-t", str(minutes * 60 + 300))
    if rc != 0:
        return jsonify(ok=False, error=out or "push failed"), 500

    rset("topic", topic); rset("started", time.time())
    rset("minutes", minutes); rset("cpu", cpu); rset("link", None)
    rset("cache_ready", use_cache)
    return jsonify(ok=True, topic=topic, pushed_at=time.time(),
                   use_cache=use_cache, minutes=minutes, cpu=cpu)

@app.get("/api/status")
def api_status():
    topic = rget("topic"); started = rget("started"); link = rget("link")
    minutes = rget("minutes", DEFAULT_MINUTES); cpu = rget("cpu", False)
    use_cache = bool(rget("cache_ready", False))

    cached_at = rget("kaggle_at", 0)
    if time.time() - cached_at > 20:
        s = run_state()
        rset("kaggle_state", s); rset("kaggle_at", time.time())
    else:
        s = rget("kaggle_state")

    active = is_active(s)
    last_msg = None; last_age = None
    if topic:
        msgs = ntfy_read(topic)
        if msgs:
            last = msgs[-1]
            last_msg = last.get("message", "")
            last_age = int(time.time() - last.get("time", time.time()))
        new_link = find_link(msgs)
        if new_link and new_link != link:
            link = new_link; rset("link", link)

    expires_in = None
    if started and active:
        expires_in = max(0, int(started + minutes * 60 - time.time()))

    return jsonify(ok=True, kaggle_state=s, active=active,
                   link=link if active else None,
                   last_link=link, last_msg=last_msg, last_msg_age=last_age,
                   started=started, minutes=minutes, cpu=cpu,
                   use_cache=use_cache, expires_in=expires_in,
                   has_run=bool(topic), now=time.time())

@app.post("/api/stop")
def api_stop():
    topic = rget("topic")
    if not topic:
        return jsonify(ok=False, error="No saved run."), 400
    try: ntfy_send(topic, "STOP")
    except Exception as e: return jsonify(ok=False, error=f"STOP send failed: {e}"), 500
    return jsonify(ok=True, message="Stop sent. Kernel checks every 15 seconds.")

@app.post("/api/forget")
def api_forget():
    for k in ("topic","started","link","minutes","cpu","kaggle_state","kaggle_at","last_msg"):
        rdel(k)
    return jsonify(ok=True)

@app.post("/api/cache")
def api_cache():
    s = run_state(CACHE_ID)
    if is_active(s):
        return jsonify(ok=False, error="A cache run is already active.", state=s)
    write_kernel(BUILD / "cache", CACHE_ID, CACHE_SLUG, CACHE_TEMPLATE, False, [], [])
    rc, out = kaggle("kernels", "push", "-p", str(BUILD / "cache"), "-t", "10800")
    if rc != 0: return jsonify(ok=False, error=out or "push failed"), 500
    return jsonify(ok=True, message="Cache kernel pushed (CPU only, no GPU hours).")

@app.get("/api/cache_status")
def api_cache_status():
    s = run_state(CACHE_ID)
    if s == "COMPLETE": rset("cache_ready", True)
    return jsonify(ok=True, state=s, ready=bool(rget("cache_ready", False)))

# ---------- Web UI ----------
PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RVC Remote</title>
<style>
:root{--bg:#0f0b14;--card:#1a1424;--line:#2d2440;--txt:#f1ecf8;--mute:#9a8fb0;--acc:#ff6fae;--acc2:#b57cff;--ok:#4ade80;--bad:#ff5d6c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:22px 16px 60px}
h1{margin:0 0 4px;font-size:26px;background:linear-gradient(90deg,var(--acc),var(--acc2));-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--mute);margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:14px}
.card h2{margin:0 0 12px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--mute)}
label{display:block;font-size:13px;color:var(--mute);margin:10px 0 4px}
input[type=text],input[type=number],select{width:100%;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:#120d1a;color:var(--txt);font:inherit}
input[type=range]{width:100%;accent-color:var(--acc)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.row>*{flex:1}
button{cursor:pointer;border:0;border-radius:10px;padding:10px 16px;font:inherit;font-weight:600;color:#fff;background:linear-gradient(90deg,var(--acc),var(--acc2))}
button.ghost{background:transparent;border:1px solid var(--line);color:var(--txt);font-weight:500}
button:disabled{opacity:.45;cursor:not-allowed}
.pill{display:inline-block;padding:3px 10px;border-radius:99px;font-size:12px;border:1px solid var(--line);color:var(--mute);float:right}
.pill.ok{color:var(--ok);border-color:var(--ok)}
.pill.bad{color:var(--bad);border-color:var(--bad)}
.val{float:right;color:var(--txt);font-variant-numeric:tabular-nums}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.chips button{padding:6px 12px;font-size:13px}
.msg{margin-top:10px;font-size:14px;min-height:20px}
.msg.err{color:var(--bad)}.msg.ok{color:var(--ok)}
.check{display:flex;gap:8px;align-items:center;margin:8px 0;font-size:14px}
.check input{accent-color:var(--acc)}
audio{width:100%;margin-top:8px}
details summary{cursor:pointer;color:var(--mute);margin:8px 0}
.res{border-top:1px solid var(--line);padding-top:12px;margin-top:12px}
.res small{color:var(--mute)}
a.dl{color:var(--acc);text-decoration:none;font-size:14px}
.mono{font-family:monospace;word-break:break-all;font-size:13px}
</style></head><body><div class="wrap">
<h1>RVC Remote</h1>
<p class="sub">Render remote control + direct browser voice conversion.</p>

<div class="card">
  <h2>Server <span id="spill" class="pill">checking…</span></h2>
  <div class="row">
    <button id="start">Start GPU</button>
    <button id="startcpu" class="ghost">Start CPU</button>
    <button id="stop" class="ghost">Stop</button>
    <button id="forget" class="ghost" style="flex:0 0 auto">Reset</button>
  </div>
  <label>Run length (minutes)</label>
  <input type="number" id="minutes" value="60" min="5" max="540">
  <div id="smsg" class="msg"></div>
</div>

<div class="card">
  <h2>Kaggle kernel <span id="kpill" class="pill">—</span></h2>
  <div id="kinfo" class="mono" style="color:var(--mute)">Not started.</div>
  <div id="heartbeat" class="msg"></div>
</div>

<div class="card" id="linkcard" style="display:none">
  <h2>Gradio link</h2>
  <div id="linkbox" class="mono" style="color:var(--acc)"></div>
  <div class="row" style="margin-top:10px">
    <button id="connect">Connect to voice UI</button>
  </div>
  <div id="cmsg" class="msg"></div>
</div>

<div class="card" id="uicard" style="display:none">
  <h2>Voice conversion</h2>
  <label for="model">Voice model</label>
  <select id="model"></select>
  <label for="index">Index file</label>
  <select id="index"></select>

  <label>Your audio</label>
  <div class="row">
    <input type="file" id="file" accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.webm">
    <button class="ghost" id="rec" style="flex:0 0 auto">Record</button>
  </div>
  <div id="recinfo" class="msg"></div>
  <audio id="preview" controls style="display:none"></audio>

  <label>Pitch <span class="val" id="pitchv">10</span></label>
  <input type="range" id="pitch" min="-24" max="24" step="1" value="10">
  <div class="chips">
    <button class="ghost" data-p="8">+8</button>
    <button class="ghost" data-p="10">+10</button>
    <button class="ghost" data-p="12">+12</button>
    <button class="ghost" data-p="0">0</button>
  </div>
  <label>Index rate <span class="val" id="irv">0.30</span></label>
  <input type="range" id="ir" min="0" max="1" step="0.05" value="0.3">
  <label>Pitch method</label>
  <select id="f0">
    <option>rmvpe</option><option>crepe</option><option>crepe-tiny</option><option>fcpe</option>
  </select>
  <details><summary>More settings</summary>
    <label>Volume <span class="val" id="volv">1.00</span></label>
    <input type="range" id="vol" min="0" max="1" step="0.05" value="1">
    <label>Protect <span class="val" id="prov">0.33</span></label>
    <input type="range" id="pro" min="0" max="0.5" step="0.01" value="0.33">
    <label>Format</label>
    <select id="fmt"><option>WAV</option><option>MP3</option><option>FLAC</option><option>OGG</option><option>M4A</option></select>
    <div class="check"><input type="checkbox" id="split"><span>Split audio</span></div>
    <div class="check"><input type="checkbox" id="autotune"><span>Autotune</span></div>
    <div class="check"><input type="checkbox" id="clean"><span>Clean audio</span></div>
  </details>

  <button id="go" style="width:100%;margin-top:12px" disabled>Convert</button>
  <div id="gmsg" class="msg"></div>
  <div id="results"></div>
</div>
</div>

<script type="module">
const $ = id => document.getElementById(id);
const msg = (el, t, c) => { el.textContent = t || ""; el.className = "msg " + (c || ""); };

// -------- render side: polling loop --------
let link = null, client = null, connected = false, blob = null, blobName = "audio.wav", busy = false;
let S = {save:null, convert:null, choices:null, defaults:null, labels:null};

async function api(path, opts){
  const r = await fetch(path, opts);
  let j; try { j = await r.json(); } catch(e){ j = {ok:false, error: await r.text()}; }
  return j;
}

async function poll(){
  try{
    const j = await api("/api/status");
    if(!j.ok) throw new Error(j.error || "status failed");
    const sp = $("spill");
    if(!j.has_run){ sp.textContent="idle"; sp.className="pill"; }
    else if(j.active){ sp.textContent="running"; sp.className="pill ok"; }
    else { sp.textContent = j.kaggle_state || "?"; sp.className = "pill bad"; }

    const kp = $("kpill");
    kp.textContent = j.kaggle_state || "—";
    kp.className = "pill " + (j.active ? "ok" : (j.kaggle_state ? "bad" : ""));

    const parts = [];
    if(j.started) parts.push("started " + Math.floor((j.now - j.started)/60) + " min ago");
    if(j.expires_in != null) parts.push("expires in " + Math.floor(j.expires_in/60) + " min");
    parts.push("cache: " + (j.use_cache ? "yes" : "no"));
    $("kinfo").textContent = j.has_run ? parts.join(" · ") : "Not started.";

    if(j.last_msg) msg($("heartbeat"), j.last_msg + "  (" + j.last_msg_age + "s ago)",
                       j.last_msg_age > 120 ? "err" : "");
    else msg($("heartbeat"), "");

    if(j.link && j.link !== link){
      link = j.link;
      $("linkbox").textContent = link;
      $("linkcard").style.display = "block";
      if(!connected) autoConnect();
    }
    if(!j.active){
      $("linkcard").style.display = "none";
      $("uicard").style.display = "none";
      connected = false; client = null; link = null;
      S = {save:null, convert:null, choices:null, defaults:null, labels:null};
      refreshGo();
    }
  }catch(e){
    $("spill").textContent = "offline"; $("spill").className = "pill bad";
    msg($("smsg"), e.message, "err");
  }
}
setInterval(poll, 5000); poll();

// -------- start / stop --------
$("start").onclick = () => start(false);
$("startcpu").onclick = () => start(true);
async function start(cpu){
  const minutes = parseInt($("minutes").value || "60");
  msg($("smsg"), "Pushing kernel to Kaggle…");
  $("start").disabled = $("startcpu").disabled = true;
  try{
    const j = await api("/api/start", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({cpu, minutes})
    });
    if(!j.ok) throw new Error(j.error);
    msg($("smsg"), "Pushed. Waiting for the link to appear…", "ok");
    poll();
  }catch(e){ msg($("smsg"), e.message, "err"); }
  $("start").disabled = $("startcpu").disabled = false;
}
$("stop").onclick = async () => {
  msg($("smsg"), "Sending STOP…");
  const j = await api("/api/stop", {method:"POST"});
  msg($("smsg"), j.ok ? j.message : j.error, j.ok ? "ok" : "err");
};
$("forget").onclick = async () => {
  if(!confirm("Reset Render state? The Kaggle kernel will keep running if it is active.")) return;
  await api("/api/forget", {method:"POST"});
  link = null; connected = false; client = null; S = {save:null,convert:null,choices:null,defaults:null,labels:null};
  $("linkcard").style.display = "none"; $("uicard").style.display = "none";
  msg($("smsg"), "Reset.", "ok"); poll();
};

// -------- gradio connect + api discovery --------
const N_PARAMS = 60;
const I = {terms:0, pitch:1, index_rate:2, volume:3, protect:4, f0:5, audio:6, output:7,
           model:8, index:9, split:10, autotune:11, clean:15, clean_strength:16, fmt:17, sid:59};
const F0_METHODS = ["crepe","crepe-tiny","rmvpe","fcpe"];
const FORMATS = ["WAV","MP3","FLAC","OGG","M4A"];

const asValue = x => (x && typeof x === "object" && !Array.isArray(x) && !(x instanceof Blob))
  ? (x.value ?? x.path ?? x.url) : x;

function asChoices(x){
  let ch = (x && typeof x === "object" && !Array.isArray(x)) ? x.choices : x;
  if(!Array.isArray(ch)) return [];
  return ch.map(c => Array.isArray(c) && c.length === 2
    ? {label:String(c[0]), value:c[1]}
    : {label:String(c), value:c});
}

async function autoConnect(){
  if(!link) return;
  msg($("cmsg"), "Connecting to Gradio…");
  $("connect").disabled = true;
  try{
    const mod = await import("https://cdn.jsdelivr.net/npm/@gradio/client@1/+esm");
    client = await mod.Client.connect(link);
    const info = await client.view_api();
    const eps = info.named_endpoints || {};
    const found = {convert:null, save:null, choices:null};
    for(const [name, ep] of Object.entries(eps)){
      const params = ep.parameters || [];
      const base = name.replace(/^\//,"");
      if(base.startsWith("enforce_terms") && params.length === N_PARAMS){
        const p5 = params[I.f0] || {};
        if(F0_METHODS.includes(p5.parameter_default)) found.convert = name;
      } else if(base.startsWith("save_to_wav2") && params.length === 1){
        found.save = name;
      } else if(base.startsWith("change_choices") && !found.choices){
        found.choices = name;
      }
    }
    if(!found.convert || !found.save)
      throw new Error("Connected, but the API shape is not Applio inference.");
    const params = eps[found.convert].parameters;
    S.defaults = params.map(p => p.parameter_has_default ? p.parameter_default : null);
    S.labels   = params.map(p => p.label || p.parameter_name);
    S.convert  = found.convert; S.save = found.save; S.choices = found.choices;

    // fetch choices
    let models = [], indexes = [], defModel = S.defaults[I.model], defIndex = S.defaults[I.index];
    if(S.choices){
      try{
        const r = await client.predict(S.choices, [defModel || ""]);
        models = asChoices(r.data[0]); indexes = asChoices(r.data[1]);
      }catch(e){}
    }
    for(const [key, val] of [["models", defModel], ["indexes", defIndex]]){
      if(val && !eval(key).some(c => c.value === val)){
        (key === "models" ? models : indexes).unshift({
          label: String(val).split("/").pop().replace(/\.[^.]+$/,""), value: val
        });
      }
    }
    fill($("model"), models, defModel); fill($("index"), indexes, defIndex);

    connected = true;
    $("uicard").style.display = "block";
    msg($("cmsg"), "Connected.", "ok");
  }catch(e){
    msg($("cmsg"), "Connect failed: " + e.message, "err");
  }
  $("connect").disabled = false;
  refreshGo();
}
$("connect").onclick = autoConnect;

function fill(sel, items, chosen){
  sel.innerHTML = "";
  if(!items.length){ sel.innerHTML = '<option value="">none found</option>'; return; }
  items.forEach(c => {
    const o = document.createElement("option");
    o.value = c.value; o.textContent = c.label;
    if(c.value === chosen) o.selected = true;
    sel.appendChild(o);
  });
}

// -------- sliders + recording --------
[["pitch","pitchv",0],["ir","irv",2],["vol","volv",2],["pro","prov",2]].forEach(([id,out,d])=>{
  const el = $(id), show = () => $(out).textContent = (+el.value).toFixed(d);
  el.addEventListener("input", show); show();
});
document.querySelectorAll("[data-p]").forEach(b => b.onclick = () => {
  $("pitch").value = b.dataset.p; $("pitch").dispatchEvent(new Event("input"));
});

function setAudio(b, name){
  blob = b; blobName = name;
  const p = $("preview"); p.src = URL.createObjectURL(b); p.style.display = "block";
  refreshGo();
}
$("file").onchange = e => { const f = e.target.files[0]; if(f) setAudio(f, f.name); };

function encodeWav(samples, rate){
  const buf = new ArrayBuffer(44 + samples.length*2), v = new DataView(buf);
  const w = (o,s) => { for(let i=0;i<s.length;i++) v.setUint8(o+i, s.charCodeAt(i)); };
  w(0,"RIFF"); v.setUint32(4,36+samples.length*2,true); w(8,"WAVE"); w(12,"fmt ");
  v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,1,true);
  v.setUint32(24,rate,true); v.setUint32(28,rate*2,true); v.setUint16(32,2,true); v.setUint16(34,16,true);
  w(36,"data"); v.setUint32(40,samples.length*2,true);
  let o = 44;
  for(let i=0;i<samples.length;i++,o+=2){
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s<0 ? s*0x8000 : s*0x7FFF, true);
  }
  return new Blob([v], {type:"audio/wav"});
}
let mr = null, chunks = [], timer = null, t0 = 0;
$("rec").onclick = async () => {
  if(mr && mr.state === "recording"){ mr.stop(); return; }
  try{
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    chunks = []; mr = new MediaRecorder(stream);
    mr.ondataavailable = e => chunks.push(e.data);
    mr.onstop = async () => {
      clearInterval(timer); stream.getTracks().forEach(t => t.stop());
      $("rec").textContent = "Record"; msg($("recinfo"), "Processing recording…");
      try{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const buf = await ctx.decodeAudioData(await new Blob(chunks).arrayBuffer());
        setAudio(encodeWav(buf.getChannelData(0), buf.sampleRate), "recording.wav");
        msg($("recinfo"), "Recorded " + buf.duration.toFixed(1) + " s.", "ok");
      }catch(e){ msg($("recinfo"), "Could not read recording: " + e.message, "err"); }
    };
    mr.start(); t0 = Date.now(); $("rec").textContent = "Stop";
    timer = setInterval(() => {
      const s = (Date.now()-t0)/1000;
      $("recinfo").innerHTML = '<span style="color:var(--bad)">recording ' + s.toFixed(0) + ' s</span>';
      if(s >= 60) mr.stop();
    }, 250);
  }catch(e){ msg($("recinfo"), "Mic not available: " + e.message, "err"); }
};

// -------- conversion --------
function refreshGo(){ $("go").disabled = !(connected && blob && !busy); }

$("go").onclick = async () => {
  if(!connected || !blob) return;
  busy = true; refreshGo();
  const start = Date.now();
  const tick = setInterval(() => msg($("gmsg"), "Converting… " + Math.round((Date.now()-start)/1000) + "s"), 500);
  try{
    // 1) upload via save endpoint
    const up = await client.predict(S.save, [blob]);
    const serverAudio = asValue(up.data[0]);
    const serverOut   = asValue(up.data[1]);

    // 2) build 60-arg call
    const args = [...S.defaults];
    args[I.terms] = true;
    args[I.audio] = serverAudio;
    args[I.output] = serverOut;
    args[I.pitch] = parseInt($("pitch").value);
    args[I.index_rate] = parseFloat($("ir").value);
    args[I.volume] = parseFloat($("vol").value);
    args[I.protect] = parseFloat($("pro").value);
    if(F0_METHODS.includes($("f0").value)) args[I.f0] = $("f0").value;
    if(FORMATS.includes($("fmt").value)) args[I.fmt] = $("fmt").value;
    args[I.split]    = $("split").checked;
    args[I.autotune] = $("autotune").checked;
    args[I.clean]    = $("clean").checked;
    if($("model").value) args[I.model] = $("model").value;
    if($("index").value) args[I.index] = $("index").value;

    // 3) convert
    const res = await client.predict(S.convert, args);
    const message = res.data[0];
    let out = res.data[1];
    let url = null;
    if(out && typeof out === "object"){
      if(out.url) url = out.url;
      else if(out.path) url = out.path;
      else if(out instanceof Blob || out instanceof File) url = URL.createObjectURL(out);
    } else if(typeof out === "string" && /^https?:/.test(out)) {
      url = out;
    } else if(typeof out === "string" && out.startsWith("/")) {
      // server path — try to prefix with the share link origin
      url = link.replace(/\/$/, "") + "/file=" + out;
    }
    if(!url) throw new Error(message || "No audio returned.");

    const d = document.createElement("div"); d.className = "res";
    const info = 'pitch ' + ($("pitch").value>0?"+":"") + $("pitch").value +
                 ' · ' + $("f0").value + ' · index ' + parseFloat($("ir").value).toFixed(2) +
                 ' · ' + Math.round((Date.now()-start)/1000) + 's';
    d.innerHTML = '<small>' + info + '</small>' +
      '<audio controls src="' + url + '"></audio>' +
      '<div style="margin-top:6px"><a class="dl" href="' + url + '" download>Download</a></div>';
    $("results").prepend(d);
    msg($("gmsg"), message || "Done.", "ok");
  }catch(e){
    msg($("gmsg"), "Conversion failed: " + (e.message || e), "err");
  }
  clearInterval(tick); busy = false; refreshGo();
};
</script>
</body></html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
