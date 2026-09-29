"""
app.py: voice changer backend (Render or anywhere).

RUN IT WITH ONE WORKER (state lives in memory) AND SEVERAL THREADS:
    gunicorn app:app --workers 1 --threads 16 --timeout 300

Required environment variables:
    KAGGLE_USERNAME, KAGGLE_KEY
    NGROK_AUTHTOKEN_1, NGROK_AUTHTOKEN_2
    RVC_NTFY_TOPIC_A, RVC_NTFY_TOPIC_B, RVC_NTFY_TOPIC_GPU

Optional (defaults in brackets):
    KERNEL_SLUG [rvc-gpu-server]   CACHE_SLUG [rvc-cache]
    DATASET [supporttopal/sweet-female-rvc]   MODEL_NAME [sweet_female]
    VOICE_MODEL / VOICE_INDEX      ALLOWED_ORIGIN [*]
    CPU_RUN_MINUTES [600]  CPU_SPARE_PREP_MIN [570]  CPU_KILL_OLD_MIN [600]
    GPU_RUN_MINUTES [40]   GPU_MAX_EXTENDED_MIN [90]  GPU_DAILY_RUNS [6]
    BOOT_TIMEOUT_MIN [20]        force kill + restart if not online by then
    HEARTBEAT_TIMEOUT [180]      seconds without a heartbeat = dead
    NTFY_URL [https://ntfy.sh]   NTFY_HISTORY_SECONDS [43200]
    CONVERT_WAIT_SECONDS [180]   how long a user may wait in the queue
    CONVERT_TIMEOUT [600]        max seconds for one conversion
    MAX_UPLOAD_MB [50]           RESULT_TTL_SECONDS [1800]
    ADMIN_TOKEN []               enables /api/admin/* and /api/stop (header X-Admin-Token)

Design:
  * ONE supervisor thread does all the network work (ntfy, kaggle, ngrok probe,
    push, kill). API requests only read an in-memory snapshot, so any number of
    users can poll /api/status without touching ntfy or kaggle.
  * The server starts itself. The front end can never start the CPU server.
    The only front end trigger is /api/upgrade-gpu.
  * A session only counts as alive if it has a fresh heartbeat (or a live tunnel).
    Kaggle CANCELACKNOWLEDGED/COMPLETE/ERROR, a finished message, a lost
    heartbeat, a dead tunnel or 20 min of booting all trigger kill + restart.
"""
import json
import mimetypes
import os
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from gradio_client import Client, handle_file


# ---------- configuration ----------

def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "supporttopal")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY")
KERNEL_SLUG = os.environ.get("KERNEL_SLUG", "rvc-gpu-server")
CACHE_SLUG = os.environ.get("CACHE_SLUG", "rvc-cache")
DATASET = os.environ.get("DATASET", "supporttopal/sweet-female-rvc")
MODEL_NAME = os.environ.get("MODEL_NAME", "sweet_female")
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
NTFY = os.environ.get("NTFY_URL", "https://ntfy.sh").rstrip("/")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

VOICE_MODEL = os.environ.get("VOICE_MODEL", "logs/sweet_female/sweet_female.pth")
VOICE_INDEX = os.environ.get("VOICE_INDEX", "logs/sweet_female/sweet_female.index")

TOPIC_A = os.environ.get("RVC_NTFY_TOPIC_A")
TOPIC_B = os.environ.get("RVC_NTFY_TOPIC_B")
TOPIC_GPU = os.environ.get("RVC_NTFY_TOPIC_GPU")
NGROK_TOKEN_1 = os.environ.get("NGROK_AUTHTOKEN_1")
NGROK_TOKEN_2 = os.environ.get("NGROK_AUTHTOKEN_2")

CPU_RUN_MINUTES = env_int("CPU_RUN_MINUTES", 600)
CPU_SPARE_PREP_MIN = env_int("CPU_SPARE_PREP_MIN", 570)
CPU_KILL_OLD_MIN = env_int("CPU_KILL_OLD_MIN", 600)
GPU_RUN_MINUTES = env_int("GPU_RUN_MINUTES", 40)
GPU_MAX_EXTENDED_MIN = env_int("GPU_MAX_EXTENDED_MIN", 90)
GPU_DAILY_RUNS = env_int("GPU_DAILY_RUNS", 6)
BOOT_TIMEOUT_MIN = env_int("BOOT_TIMEOUT_MIN", 20)
HEARTBEAT_TIMEOUT = env_int("HEARTBEAT_TIMEOUT", 180)
NTFY_HISTORY_SECONDS = env_int("NTFY_HISTORY_SECONDS", 43200)
CONVERT_WAIT_SECONDS = env_int("CONVERT_WAIT_SECONDS", 180)
CONVERT_TIMEOUT = env_int("CONVERT_TIMEOUT", 600)
MAX_UPLOAD_MB = env_int("MAX_UPLOAD_MB", 50)
RESULT_TTL_SECONDS = env_int("RESULT_TTL_SECONDS", 1800)

# internal tuning
SUPERVISOR_INTERVAL = 10      # seconds between supervisor ticks
ACTIVE_POLL = 20              # ntfy poll interval for active slots
IDLE_POLL = 120               # ntfy poll interval for idle slots (keeps us far under ntfy limits)
KAGGLE_ACTIVE_POLL = 45
KAGGLE_IDLE_POLL = 600
PROBE_INTERVAL = 45
PROBE_FAILS_TO_DEAD = 3
PUSH_GRACE = 150              # ignore kaggle DONE states this soon after a push (old version's state)
KILL_COOLDOWN = 120
STUCK_FORCE_AFTER = 300       # push over a kernel that will not stop after this long
SKEW = 30                     # clock skew tolerance between us and ntfy

KERNEL_ID_CACHE = f"{KAGGLE_USERNAME}/{CACHE_SLUG}"
DONE = {"COMPLETE", "ERROR", "CANCELACKNOWLEDGED", "CANCELREQUESTED"}
WORK = Path(tempfile.mkdtemp(prefix="rvc_backend_"))
TOKEN_1, TOKEN_2 = "1", "2"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


if not KAGGLE_USERNAME or not KAGGLE_KEY:
    log("[config] FATAL: KAGGLE_USERNAME and KAGGLE_KEY must both be set.")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGIN}})


# ---------- kernel template (runs on Kaggle) ----------

KERNEL_TEMPLATE = r"""
import glob, json, os, re, shutil, socket, subprocess, sys, threading, time, urllib.request

TOPIC = "__TOPIC__"
MAX_SECONDS = __MINUTES__ * 60
USE_CACHE = __USE_CACHE__
MODEL_NAME = "__MODEL_NAME__"
NGROK_TOKEN = "__NGROK__"
APPLIO = "/kaggle/working/Applio"
NTFY = "__NTFY__"
started = time.time()
ENV = dict(os.environ, PYTHONUNBUFFERED="1")
proc = None
PUBLIC_URL = None


def say(msg):
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"{NTFY}/{TOPIC}", data=msg.encode(), method="POST")
            urllib.request.urlopen(req, timeout=20).read()
            return
        except Exception as e:
            print("ntfy post failed:", e, flush=True)
            time.sleep(2 * (attempt + 1))


def sh(cmd, cwd=None):
    print(">>", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=ENV)


def stop_requested():
    try:
        url = f"{NTFY}/{TOPIC}/json?poll=1&since={int(started)}"
        with urllib.request.urlopen(url, timeout=20) as r:
            for raw in r.read().decode("utf-8", "replace").splitlines():
                try:
                    m = json.loads(raw)
                except ValueError:
                    continue
                if m.get("event") == "message" and m.get("message", "").strip() == "STOP":
                    return True
    except Exception as e:
        print("poll failed:", e, flush=True)
    return False


def kill_everything(reason):
    say("STATUS " + reason)
    try:
        if proc is not None:
            proc.terminate()
    except Exception:
        pass
    try:
        import psutil
        for child in psutil.Process().children(recursive=True):
            try:
                child.kill()
            except Exception:
                pass
    except Exception:
        subprocess.run(["pkill", "-P", str(os.getpid())])
    say("STATUS finished")
    os._exit(0)


def watchdog():
    tick = 0
    while True:
        time.sleep(20)
        tick += 1
        if time.time() - started > MAX_SECONDS:
            kill_everything("time limit reached, shutting down")
        if stop_requested():
            kill_everything("stop received, shutting down")
        if tick % 3 == 0:
            say(f"STATUS alive {int((time.time() - started) / 60)} min")
        if tick % 15 == 0 and PUBLIC_URL:
            say("LINK " + PUBLIC_URL)


def wait_port(port, timeout=600):
    end = time.time() + timeout
    while time.time() < end:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError("Applio exited during startup")
        try:
            socket.create_connection(("127.0.0.1", port), 2).close()
            return
        except OSError:
            time.sleep(3)
    raise RuntimeError("Applio did not open port " + str(port))


say("STATUS started token=__TOKEN_NAME__")
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
    sh("sed -i 's/weights_only=True/weights_only=False/g' /kaggle/working/Applio/rvc/infer/infer.py")
    sh("apt-get update -y")
    sh("apt-get install -y portaudio19-dev libportaudio2")
    if libs is None:
        sh("pip install -q uv")
        sh("uv pip install -q -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match --system", cwd=APPLIO)
    say("STATUS downloading Applio models")
    sh(f"{sys.executable} core.py prerequisites --models", cwd=APPLIO)

    say("STATUS copying voice files")
    dest = f"{APPLIO}/logs/{MODEL_NAME}"
    os.makedirs(dest, exist_ok=True)
    for ext in ("pth", "index"):
        for f in glob.glob(f"/kaggle/input/**/*.{ext}", recursive=True):
            shutil.copy(f, dest)
            print("copied", f, flush=True)

    say("STATUS installing pyngrok")
    sh("pip install -q pyngrok")

    say("STATUS starting Applio")
    proc = subprocess.Popen(
        [sys.executable, "-u", "app.py", "--listen", "--port", "7860", "--client"],
        cwd=APPLIO, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    def reader():
        for line in proc.stdout:
            print(line, end="", flush=True)

    threading.Thread(target=reader, daemon=True).start()

    wait_port(7860)

    say("STATUS opening ngrok tunnel")
    from pyngrok import conf, ngrok

    conf.get_default().auth_token = NGROK_TOKEN
    conf.get_default().monitor_thread = False

    existing = ngrok.get_tunnels(conf.get_default())
    if existing:
        PUBLIC_URL = existing[0].public_url
    else:
        PUBLIC_URL = ngrok.connect(7860, bind_tls=True).public_url

    say("LINK " + PUBLIC_URL)

    while proc.poll() is None:
        time.sleep(5)
    say("STATUS Applio exited on its own")
except Exception as e:
    say(f"STATUS failed: {e}")
    raise
finally:
    if proc is not None and proc.poll() is None:
        proc.terminate()
    say("STATUS finished")
"""


# ---------- small helpers ----------

def fail(message, code=400, **extra):
    return jsonify(ok=False, error=str(message), **extra), code


class ServerOffline(Exception):
    pass


def kaggle_cli(*args, timeout=60):
    """Run the kaggle CLI with a hard timeout that kills the whole process group."""
    proc = subprocess.Popen(
        ["kaggle", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=dict(os.environ), start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, ((out or "") + (err or "")).strip()
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        return 1, f"kaggle {' '.join(args)} timed out after {timeout}s"


def parse_state(out):
    m = re.search(r'status\s+"?([\w.]+)"?', out)
    if not m:
        return None
    return m.group(1).split(".")[-1].replace("_", "").upper()


def run_state(kernel_id):
    if not shutil.which("kaggle"):
        return None
    code, out = kaggle_cli("kernels", "status", kernel_id, timeout=30)
    if code != 0:
        return None
    return parse_state(out)


def kaggle_cancel(kernel_id):
    """Best effort. Not every kaggle CLI version has a cancel command; failure is fine
    because STOP over ntfy and Kaggle's own run timeout are the fallbacks."""
    if not shutil.which("kaggle"):
        return
    rc, out = kaggle_cli("kernels", "cancel", kernel_id, timeout=30)
    log(f"[cancel] {kernel_id} rc={rc} {out[:120]}")


def is_active(state):
    return state is not None and state not in DONE


def write_kernel(kernel_id, code, gpu, build_dir):
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "script.py").write_text(code)
    meta = {
        "id": kernel_id,
        "title": kernel_id.split("/")[-1].replace("-", " "),
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if gpu else "false",
        "enable_internet": "true",
        "dataset_sources": [DATASET],
        "competition_sources": [],
        "kernel_sources": [KERNEL_ID_CACHE],
    }
    (build_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


# ---------- ntfy ----------

NTFY_BACKOFF = {"until": 0}


def ntfy_read(topic, since_ts):
    """Returns a list of messages, or None when ntfy could not be read.
    None means UNKNOWN, never 'empty'. This is what caused false 'booting' states."""
    if time.time() < NTFY_BACKOFF["until"]:
        return None
    url = f"{NTFY}/{topic}/json?poll=1&since={int(since_ts)}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            text = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code == 429:
            NTFY_BACKOFF["until"] = time.time() + 60
            log("[ntfy] rate limited (429), backing off 60s")
        return None
    except Exception:
        return None
    out = []
    for line in text.splitlines():
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("event") == "message":
            out.append(m)
    return out


def ntfy_send(topic, text, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(f"{NTFY}/{topic}", data=text.encode(), method="POST")
            urllib.request.urlopen(req, timeout=20).read()
            return True
        except Exception as e:
            log(f"[ntfy] send failed ({i + 1}/{retries}): {e}")
            time.sleep(1 + i)
    return False


# ---------- slots (only the supervisor thread mutates these) ----------

def make_slot(sid, kind, topic):
    return {
        "id": sid, "kind": kind, "topic": topic,
        "kernel_id": f"{KAGGLE_USERNAME}/{KERNEL_SLUG}-{sid.replace('_', '-')}",
        # lifecycle
        "pushed_at": 0, "token": None, "upgrade_ts": 0, "ever_live": False,
        # what ntfy says
        "link": None, "session_start": 0, "session_token": None, "ended": False,
        "last_seen": 0, "last_message": None,
        "poll_ts": 0, "seen": {}, "ntfy_ok": True, "next_poll": 0,
        # what kaggle says
        "kaggle_state": None, "kaggle_at": 0, "next_kaggle": 0,
        # tunnel probe
        "probe_fails": 0, "probe_ok_at": 0, "next_probe": 0,
        # failure handling
        "killed_at": 0, "cooldown_until": 0, "next_push_at": 0,
        "push_fails": 0, "last_error": None,
    }


SLOTS = [
    make_slot("cpu_a", "cpu", TOPIC_A),
    make_slot("cpu_b", "cpu", TOPIC_B),
    make_slot("gpu", "gpu", TOPIC_GPU),
]
CPU_SLOTS = SLOTS[:2]
GPU_SLOT = SLOTS[2]
SLOT_BY_ID = {s["id"]: s for s in SLOTS}

SNAP_LOCK = threading.Lock()
SNAPSHOT = {"value": None}
WAKE = threading.Event()
RECHECK = threading.Event()
GPU_REQUEST = {"at": 0}
ADMIN_FLAGS = {"restart": False}

UPGRADE_LOG = []
UPGRADE_LOCK = threading.Lock()


def request_recheck():
    RECHECK.set()
    WAKE.set()


def upgrades_remaining():
    with UPGRADE_LOCK:
        cutoff = time.time() - 24 * 3600
        UPGRADE_LOG[:] = [t for t in UPGRADE_LOG if t > cutoff]
        return GPU_DAILY_RUNS - len(UPGRADE_LOG)


def record_upgrade(ts):
    with UPGRADE_LOCK:
        UPGRADE_LOG.append(ts)


def refund_upgrade(ts):
    with UPGRADE_LOCK:
        if ts in UPGRADE_LOG:
            UPGRADE_LOG.remove(ts)


# ---------- reading state ----------

def apply_messages(slot, msgs):
    seen = slot["seen"]
    for m in msgs:
        mid = m.get("id")
        t = int(m.get("time", 0) or 0)
        if mid in seen:
            continue
        seen[mid] = t
        if t > slot["poll_ts"]:
            slot["poll_ts"] = t
        text = (m.get("message") or "").strip()
        if text == "STOP":
            continue
        if text == "STATUS reset":          # tombstone posted by the backend
            slot.update(link=None, session_start=0, session_token=None,
                        ended=False, last_seen=0, last_message=None)
            continue
        if text.startswith("STATUS started"):
            mm = re.search(r"token=(\d)", text)
            slot.update(link=None, session_start=t, ended=False,
                        session_token=mm.group(1) if mm else None)
        elif text == "STATUS finished":
            slot.update(link=None, ended=True)
        else:
            if slot["ended"] or not slot["session_start"]:   # older kernels have no 'started'
                slot.update(session_start=t, ended=False)
            if text.startswith("LINK "):
                slot["link"] = text[5:].strip()
        slot["last_seen"] = t
        slot["last_message"] = text
    horizon = slot["poll_ts"] - 10
    for k in [k for k, v in seen.items() if v < horizon]:
        del seen[k]


def refresh_ntfy(slot, now):
    if not slot["topic"] or now < slot["next_poll"]:
        return
    active = slot["pushed_at"] or slot["link"]
    slot["next_poll"] = now + (ACTIVE_POLL if active else IDLE_POLL)
    since = (slot["poll_ts"] - 2) if slot["poll_ts"] else (now - NTFY_HISTORY_SECONDS)
    msgs = ntfy_read(slot["topic"], since)
    if msgs is None:
        slot["ntfy_ok"] = False
        return
    slot["ntfy_ok"] = True
    apply_messages(slot, msgs)


def refresh_kaggle(slot, now):
    if now < slot["next_kaggle"]:
        return
    active = slot["pushed_at"] or slot["link"]
    slot["next_kaggle"] = now + (KAGGLE_ACTIVE_POLL if active else KAGGLE_IDLE_POLL)
    st = run_state(slot["kernel_id"])
    if st is not None:
        slot["kaggle_state"] = st
        slot["kaggle_at"] = now


def probe_tunnel(url):
    """True = reachable, False = ngrok says offline, None = unknown."""
    req = urllib.request.Request(url, headers={
        "ngrok-skip-browser-warning": "1", "User-Agent": "voice-backend-probe"})
    try:
        with urllib.request.urlopen(req, timeout=8):
            return True
    except urllib.error.HTTPError as e:
        return False if e.headers.get("ngrok-error-code") else True
    except Exception:
        return None


def refresh_probe(slot, now):
    if not slot["link"] or slot["ended"] or now < slot["next_probe"]:
        return
    if not current_session(slot):
        return
    slot["next_probe"] = now + PROBE_INTERVAL
    res = probe_tunnel(slot["link"])
    if res is True:
        slot["probe_fails"] = 0
        slot["probe_ok_at"] = now
    elif res is False:
        slot["probe_fails"] += 1


def current_session(slot):
    """True when the messages we see belong to the session we pushed (not an old one)."""
    if not slot["session_start"]:
        return False
    return slot["session_start"] >= slot["pushed_at"] - SKEW


def link_alive(slot, now):
    if not slot["link"] or slot["ended"] or not current_session(slot):
        return False
    if slot["probe_fails"] >= PROBE_FAILS_TO_DEAD:
        return False
    stale = (slot["ntfy_ok"] and slot["last_seen"]
             and now - slot["last_seen"] > HEARTBEAT_TIMEOUT)
    if stale and not (now - slot["probe_ok_at"] < 3 * PROBE_INTERVAL):
        return False
    return True


def slot_public_state(slot, now):
    if link_alive(slot, now):
        return "ready"
    return "booting" if slot["pushed_at"] else "idle"


# ---------- tokens ----------

def token_value(name):
    return NGROK_TOKEN_1 if name == TOKEN_1 else NGROK_TOKEN_2


def claim_token(slot, preferred=None):
    held = {s["token"] for s in SLOTS if s is not slot and s["token"]}
    if preferred and preferred not in held:
        slot["token"] = preferred
        return preferred
    for t in (TOKEN_1, TOKEN_2):
        if t not in held:
            slot["token"] = t
            return t
    return None


# ---------- push / kill ----------

def push_allowed(slot, now):
    return now >= slot["cooldown_until"] and now >= slot["next_push_at"]


def push_slot(slot, minutes, gpu, preferred_token=None):
    now = time.time()

    def failed(msg, backoff=30):
        slot["push_fails"] += 1
        slot["next_push_at"] = time.time() + min(300, backoff * slot["push_fails"])
        slot["last_error"] = msg
        log(f"[push] {slot['id']} not pushed: {msg}")
        return False, {"error": msg}

    if not NGROK_TOKEN_1 or not NGROK_TOKEN_2:
        return failed("Both NGROK_AUTHTOKEN_1 and NGROK_AUTHTOKEN_2 must be set.", 60)
    if not KAGGLE_KEY:
        return failed("KAGGLE_KEY is not set on the server.", 60)
    if not slot["topic"]:
        return failed(f"Topic for slot {slot['id']} is not set.", 60)
    if not shutil.which("kaggle"):
        return failed("The kaggle command is not available on this server.", 60)

    st = run_state(slot["kernel_id"])
    if st is not None:
        slot["kaggle_state"], slot["kaggle_at"] = st, time.time()
    if is_active(st):
        if slot["killed_at"] == 0:
            return failed("kernel already active, waiting to adopt it", 10)
        if now - slot["killed_at"] < STUCK_FORCE_AFTER:
            kaggle_cancel(slot["kernel_id"])
            return failed("previous session still stopping", 15)
        log(f"[push] {slot['id']} pushing over a kernel that would not stop")

    token = claim_token(slot, preferred_token)
    if not token:
        return failed("no free ngrok token right now", 30)

    ntfy_send(slot["topic"], "STATUS reset", retries=2)   # clear any stale link
    code = (KERNEL_TEMPLATE
            .replace("__TOPIC__", slot["topic"])
            .replace("__MINUTES__", str(minutes))
            .replace("__USE_CACHE__", "True")
            .replace("__MODEL_NAME__", MODEL_NAME)
            .replace("__NTFY__", NTFY)
            .replace("__TOKEN_NAME__", token)
            .replace("__NGROK__", token_value(token) or ""))
    build_dir = WORK / "kernel_build" / slot["id"]
    write_kernel(slot["kernel_id"], code, gpu=gpu, build_dir=build_dir)
    rc, out = kaggle_cli("kernels", "push", "-p", str(build_dir),
                         "-t", str(minutes * 60 + 300), timeout=180)
    if rc != 0:
        slot["token"] = None
        return failed(f"Kaggle push failed: {out[:300]}", 30)

    t = time.time()
    slot.update(pushed_at=int(t), token=token, link=None, session_start=0,
                session_token=None, ended=False, last_seen=0, last_message=None,
                ever_live=False, probe_fails=0, probe_ok_at=0, next_probe=0,
                kaggle_state=None, kaggle_at=0, next_kaggle=t + PUSH_GRACE,
                next_poll=t + 10, push_fails=0, next_push_at=0, last_error=None)
    log(f"[push] ok {slot['id']} token={token}")
    request_recheck()
    return True, {"kaggle_output": out, "token": token}


def kill_slot(slot, reason):
    now = time.time()
    log(f"[kill] {slot['id']}: {reason}")
    was_gpu_never_live = slot["kind"] == "gpu" and not slot["ever_live"]
    if slot["topic"]:
        ntfy_send(slot["topic"], "STOP")
    kaggle_cancel(slot["kernel_id"])
    if slot["topic"]:
        ntfy_send(slot["topic"], "STATUS reset")
    if was_gpu_never_live and slot["upgrade_ts"]:
        refund_upgrade(slot["upgrade_ts"])
    slot.update(pushed_at=0, token=None, upgrade_ts=0, ever_live=False, link=None,
                session_start=0, session_token=None, ended=False, last_seen=0,
                last_message=None, probe_fails=0, probe_ok_at=0, next_probe=0,
                killed_at=now, cooldown_until=now + KILL_COOLDOWN,
                kaggle_state=None, kaggle_at=0, next_kaggle=now + 15,
                next_poll=now + 5, last_error=reason)
    CLIENT_CACHE.update(url=None, client=None, endpoints=None)


def maybe_adopt(slot, now):
    """After a backend restart the slots are empty; pick up whatever is really running."""
    if slot["pushed_at"] or now < slot["cooldown_until"]:
        return
    if link_alive(slot, now):
        slot["pushed_at"] = int(slot["session_start"] or now)
        tok = slot["session_token"]
        held = {s["token"] for s in SLOTS if s is not slot and s["token"]}
        slot["token"] = tok if tok and tok not in held else claim_token(slot)
        slot["ever_live"] = True
        log(f"[adopt] {slot['id']} live session")
    elif (is_active(slot["kaggle_state"]) and slot["kaggle_at"] > slot["killed_at"]
          and (slot["killed_at"] == 0 or now - slot["killed_at"] > 600)):
        slot["pushed_at"] = int(now)
        claim_token(slot)
        log(f"[adopt] {slot['id']} kaggle kernel running, treating as booting")


def dead_reason(slot, now):
    age = now - slot["pushed_at"]
    cur = current_session(slot)
    if cur and slot["ended"]:
        return "kernel reported finished"
    if slot["kaggle_state"] in DONE and slot["kaggle_at"] >= slot["pushed_at"] + PUSH_GRACE:
        return f"kaggle state {slot['kaggle_state']}"
    if slot["probe_fails"] >= PROBE_FAILS_TO_DEAD:
        return "tunnel offline"
    if slot["link"] and cur and not link_alive(slot, now):
        return "heartbeat lost"
    if not link_alive(slot, now) and age > BOOT_TIMEOUT_MIN * 60:
        return f"stuck booting for over {BOOT_TIMEOUT_MIN} min"
    return None


def cleanup_orphan(slot, now):
    """A link we did not push and that has no heartbeat: clear it so it can never be served."""
    if (not slot["pushed_at"] and slot["link"] and slot["ntfy_ok"]
            and not link_alive(slot, now) and now >= slot["cooldown_until"]):
        log(f"[orphan] clearing stale link on {slot['id']}")
        if slot["topic"]:
            ntfy_send(slot["topic"], "STOP")
            ntfy_send(slot["topic"], "STATUS reset")
        slot.update(link=None, session_start=0, ended=False, last_seen=0,
                    cooldown_until=now + 30, next_poll=now + 5)


# ---------- management ----------

def other_cpu(slot):
    return CPU_SLOTS[1] if slot is CPU_SLOTS[0] else CPU_SLOTS[0]


def manage_cpu(now):
    active = [s for s in CPU_SLOTS if s["pushed_at"]]
    live = [s for s in active if link_alive(s, now)]

    if len(live) == 2:
        kill_slot(min(live, key=lambda s: s["pushed_at"]), "replaced by newer CPU session")
        return

    if not active:
        cands = [s for s in CPU_SLOTS if push_allowed(s, now)]
        if cands:
            slot = min(cands, key=lambda s: s["killed_at"])
            push_slot(slot, CPU_RUN_MINUTES, gpu=False, preferred_token=TOKEN_1)
        return

    if len(live) == 1 and len(active) == 1:
        prim = live[0]
        spare = other_cpu(prim)
        age_min = (now - prim["pushed_at"]) / 60
        if age_min >= CPU_SPARE_PREP_MIN and push_allowed(spare, now):
            preferred = TOKEN_2 if prim["token"] == TOKEN_1 else TOKEN_1
            push_slot(spare, CPU_RUN_MINUTES, gpu=False, preferred_token=preferred)
        if age_min >= CPU_KILL_OLD_MIN and not spare["pushed_at"]:
            kill_slot(prim, "max age reached with no replacement")


def manage_gpu(now):
    gpu = GPU_SLOT
    cpu_live = any(link_alive(s, now) for s in CPU_SLOTS)

    if gpu["pushed_at"]:
        el = (now - gpu["pushed_at"]) / 60
        if el >= GPU_RUN_MINUTES and (cpu_live or el >= GPU_MAX_EXTENDED_MIN):
            kill_slot(gpu, "gpu window over")
        return

    if not GPU_REQUEST["at"]:
        return
    if now - GPU_REQUEST["at"] > 300:
        GPU_REQUEST["at"] = 0
        return
    if upgrades_remaining() <= 0:
        GPU_REQUEST["at"] = 0
        return
    if not push_allowed(gpu, now):
        return
    held = {s["token"] for s in CPU_SLOTS if s["token"]}
    preferred = TOKEN_2 if TOKEN_1 in held else TOKEN_1
    # kernel runs up to the extended limit; the backend ends it at GPU_RUN_MINUTES when a CPU is live
    ok, _ = push_slot(gpu, GPU_MAX_EXTENDED_MIN, gpu=True, preferred_token=preferred)
    if ok:
        gpu["upgrade_ts"] = time.time()
        record_upgrade(gpu["upgrade_ts"])
        GPU_REQUEST["at"] = 0


def build_snapshot(now):
    live = [s for s in CPU_SLOTS if link_alive(s, now)]
    booting = [s for s in CPU_SLOTS if s["pushed_at"] and s not in live]
    if live:
        cpu_primary = max(live, key=lambda s: s["pushed_at"])
    elif booting:
        cpu_primary = max(booting, key=lambda s: s["pushed_at"])
    else:
        cpu_primary = None
    cpu_spare = other_cpu(cpu_primary) if cpu_primary else CPU_SLOTS[1]

    gpu_live = link_alive(GPU_SLOT, now)
    serving = GPU_SLOT if gpu_live else (cpu_primary if live else None)
    link = serving["link"] if serving else None

    pending = [s for s in SLOTS if s["pushed_at"] and not link_alive(s, now)]
    server_state = "online" if link else ("booting" if pending else "idle")
    info = serving or cpu_primary or (pending[0] if pending else None)

    boot_minutes = None
    if server_state == "booting" and pending:
        boot_minutes = int((now - min(s["pushed_at"] for s in pending)) / 60)

    errors = [s["last_error"] for s in SLOTS if s["last_error"]]
    return {
        "kaggle_state": info["kaggle_state"] if info else None,
        "active": server_state != "idle",
        "server_state": server_state,
        "link": link,
        "primary": "gpu" if gpu_live else "cpu",
        "cpu_primary_age_min": (int((now - cpu_primary["pushed_at"]) / 60)
                                if cpu_primary and cpu_primary["pushed_at"] else None),
        "cpu_spare_state": slot_public_state(cpu_spare, now),
        "gpu_state": slot_public_state(GPU_SLOT, now),
        "gpu_requested": bool(GPU_REQUEST["at"]),
        "gpu_upgrades_remaining": upgrades_remaining(),
        "gpu_upgrades_per_day": GPU_DAILY_RUNS,
        "boot_minutes": boot_minutes,
        "boot_timeout_min": BOOT_TIMEOUT_MIN,
        "last_message": info["last_message"] if info else None,
        "last_message_age_seconds": (int(now - info["last_seen"])
                                     if info and info["last_seen"] else None),
        "last_error": errors[-1] if errors else None,
        "ntfy_ok": all(s["ntfy_ok"] for s in SLOTS),
        "cpu_a_token": SLOT_BY_ID["cpu_a"]["token"],
        "cpu_b_token": SLOT_BY_ID["cpu_b"]["token"],
        "gpu_token": GPU_SLOT["token"],
        "updated_at": int(now),
    }


def cleanup_results(now):
    with RESULTS_LOCK:
        for rid, (path, created) in list(RESULTS.items()):
            if now - created > RESULT_TTL_SECONDS:
                RESULTS.pop(rid, None)
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass


def supervisor_tick():
    now = time.time()

    if RECHECK.is_set():
        RECHECK.clear()
        for s in SLOTS:
            if s["pushed_at"] or s["link"]:
                s["next_poll"] = s["next_kaggle"] = s["next_probe"] = 0

    if ADMIN_FLAGS["restart"]:
        ADMIN_FLAGS["restart"] = False
        for s in SLOTS:
            if s["pushed_at"] or s["link"]:
                kill_slot(s, "admin restart")

    for s in SLOTS:
        refresh_ntfy(s, now)
        refresh_kaggle(s, now)
        refresh_probe(s, now)
        if s["pushed_at"] and link_alive(s, now):
            s["ever_live"] = True
        maybe_adopt(s, now)
        cleanup_orphan(s, now)

    for s in SLOTS:
        if s["pushed_at"]:
            reason = dead_reason(s, now)
            if reason:
                kill_slot(s, reason)

    now = time.time()
    manage_cpu(now)
    manage_gpu(now)
    cleanup_results(now)

    snap = build_snapshot(time.time())
    with SNAP_LOCK:
        SNAPSHOT["value"] = snap


def supervisor():
    log("[supervisor] started")
    while True:
        try:
            supervisor_tick()
        except Exception:
            log("[supervisor] exception:\n" + traceback.format_exc())
        WAKE.wait(SUPERVISOR_INTERVAL)
        WAKE.clear()


def get_snapshot():
    with SNAP_LOCK:
        return SNAPSHOT["value"]


# ---------- Applio proxy ----------

CONVERT_LOCK = threading.Lock()
CLIENT_LOCK = threading.RLock()
COUNT_LOCK = threading.Lock()
WAITING = {"n": 0}
RESULTS = {}
RESULTS_LOCK = threading.Lock()
CLIENT_CACHE = {"url": None, "client": None, "endpoints": None}

N_INFER_PARAMS = 60
N_TTS_PARAMS = 25
I_INFER = dict(terms=0, pitch=1, index_rate=2, volume=3, protect=4, f0=5, audio=6,
               output=7, model=8, index=9, split=10, autotune=11, clean=15,
               clean_strength=16, fmt=17, sid=59)
I_TTS = dict(terms=0, tts_path=1, text=2, voice=3, rate=4, pitch=5, index_rate=6,
             volume=7, protect=8, f0=9, tts_out=10, rvc_out=11, model=12, index=13,
             split=14, autotune=15, clean=19, clean_strength=20, fmt=21, sid=24)


def as_value(x):
    if isinstance(x, dict):
        return x.get("value", x.get("path"))
    return x


def parse_literal_choices(type_str):
    if not type_str or not type_str.startswith("Literal["):
        return None
    return re.findall(r"'([^']*)'", type_str) or None


def discover(client):
    info = client.view_api(return_format="dict", print_info=False)
    found = {"infer": None, "tts": None, "save": None}
    meta = {}
    for name, ep in info.get("named_endpoints", {}).items():
        params = ep.get("parameters", [])
        base = name.lstrip("/")
        if base.startswith("enforce_terms_batch"):
            continue
        if len(params) == N_INFER_PARAMS and params[I_INFER["f0"]].get("parameter_default") in (
                "crepe", "crepe-tiny", "rmvpe", "fcpe"):
            found["infer"] = name
        elif len(params) == N_TTS_PARAMS:
            found["tts"] = name
        elif base.startswith("save_to_wav2") and len(params) == 1:
            found["save"] = name
        defaults = [p.get("parameter_default") if p.get("parameter_has_default") else None for p in params]
        choices = [parse_literal_choices(p.get("python_type", {}).get("type")) for p in params]
        meta[name] = {"defaults": defaults, "choices": choices}
    if not found["infer"] or not found["save"]:
        raise RuntimeError("Connected, but this does not look like Applio's Inference API.")
    return found, meta


def get_endpoints():
    snap = get_snapshot()
    link = snap.get("link") if snap else None
    if not link:
        raise ServerOffline("The voice server is not online yet.")
    with CLIENT_LOCK:
        if CLIENT_CACHE["url"] == link and CLIENT_CACHE["client"] is not None:
            return CLIENT_CACHE["client"], CLIENT_CACHE["endpoints"], link
        client = Client(link, verbose=False)
        found, meta = discover(client)
        CLIENT_CACHE.update(url=link, client=client, endpoints=(found, meta))
        return client, (found, meta), link


def with_reconnect(fn):
    try:
        client, (found, meta), _ = get_endpoints()
        return fn(client, found, meta)
    except ServerOffline:
        raise
    except Exception as first:
        with CLIENT_LOCK:
            CLIENT_CACHE.update(url=None, client=None, endpoints=None)
        try:
            client, (found, meta), _ = get_endpoints()
            return fn(client, found, meta)
        except ServerOffline:
            raise
        except Exception:
            raise first


def run_job(client, args, api_name):
    job = client.submit(*args, api_name=api_name)
    try:
        return job.result(timeout=CONVERT_TIMEOUT)
    except Exception:
        try:
            job.cancel()
        except Exception:
            pass
        raise


def parse_settings(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("settings must be an object")
    out = {"pitch": int(cfg.get("pitch", 10)), "index_rate": float(cfg.get("index_rate", 0.3))}
    for k in ("volume", "protect", "clean_strength"):
        if k in cfg:
            out[k] = float(cfg[k])
    if "rate" in cfg:
        out["rate"] = int(cfg["rate"])
    for k in ("f0", "fmt", "voice"):
        if cfg.get(k):
            out[k] = str(cfg[k])
    for k in ("split", "autotune", "clean"):
        if k in cfg:
            out[k] = bool(cfg[k])
    return out


def acquire_converter():
    with COUNT_LOCK:
        WAITING["n"] += 1
    try:
        return CONVERT_LOCK.acquire(timeout=CONVERT_WAIT_SECONDS)
    finally:
        with COUNT_LOCK:
            WAITING["n"] -= 1


def _store_result(res):
    message, out = res[0], as_value(res[1])
    if not out or not os.path.exists(str(out)):
        return fail(message or "Applio returned no audio.", 502)
    rid = secrets.token_hex(6)
    dest = WORK / f"out_{rid}{Path(str(out)).suffix or '.wav'}"
    shutil.copy(str(out), dest)
    with RESULTS_LOCK:
        RESULTS[rid] = (dest, time.time())
    return jsonify(ok=True, id=rid, message=message)


def require_topics():
    if not TOPIC_A or not TOPIC_B or not TOPIC_GPU:
        return fail("One or more RVC_NTFY_TOPIC_* variables are missing.", 500)
    return None


def is_admin():
    return bool(ADMIN_TOKEN) and request.headers.get("X-Admin-Token") == ADMIN_TOKEN


# ---------- routes ----------

@app.get("/")
def home():
    return jsonify(ok=True, service="voice changer backend")


@app.get("/api/status")
def status():
    bad = require_topics()
    if bad:
        return bad
    snap = get_snapshot()
    with COUNT_LOCK:
        queue = WAITING["n"]
    if snap is None:
        return jsonify(ok=True, active=True, server_state="booting", link=None,
                       primary="cpu", last_message="supervisor starting",
                       gpu_upgrades_remaining=upgrades_remaining(),
                       gpu_upgrades_per_day=GPU_DAILY_RUNS, convert_queue=queue)
    return jsonify(ok=True, convert_queue=queue, **snap)


@app.post("/api/start")
def start():
    """The front end cannot start the server. The backend keeps it running by itself."""
    snap = get_snapshot() or {}
    return jsonify(ok=True, already_running=True,
                   message="Server is running (managed automatically).",
                   **{k: v for k, v in snap.items() if k != "ok"})


@app.post("/api/stop")
def stop():
    if not is_admin():
        return jsonify(ok=True, stopped=False,
                       message="The server is managed automatically and cannot be stopped from here.")
    ADMIN_FLAGS["restart"] = True
    WAKE.set()
    return jsonify(ok=True, stopped=True, message="Restart requested.")


@app.post("/api/admin/restart")
def admin_restart():
    if not is_admin():
        return fail("Forbidden.", 403)
    ADMIN_FLAGS["restart"] = True
    WAKE.set()
    return jsonify(ok=True, message="Force kill and restart requested.")


@app.get("/api/admin/debug")
def admin_debug():
    if not is_admin():
        return fail("Forbidden.", 403)
    now = time.time()
    rows = []
    for s in SLOTS:
        rows.append({k: v for k, v in s.items() if k not in ("seen", "topic")}
                    | {"alive": link_alive(s, now),
                       "age_min": int((now - s["pushed_at"]) / 60) if s["pushed_at"] else None})
    return jsonify(ok=True, slots=rows, gpu_request=GPU_REQUEST["at"])


@app.post("/api/upgrade-gpu")
def upgrade_gpu():
    if not TOPIC_GPU:
        return fail("RVC_NTFY_TOPIC_GPU is not set on the server.", 500)
    snap = get_snapshot() or {}
    if snap.get("gpu_state") in ("ready", "booting"):
        return jsonify(ok=True, already_running=True,
                       message="GPU is already running or booting.", **snap)
    if GPU_REQUEST["at"]:
        return jsonify(ok=True, already_running=True,
                       message="GPU request already queued.", **snap)
    remaining = upgrades_remaining()
    if remaining <= 0:
        return fail("Daily GPU limit reached. Try again later.", 429, gpu_upgrades_remaining=0)
    GPU_REQUEST["at"] = time.time()
    WAKE.set()
    return jsonify(ok=True, already_running=False,
                   message=f"GPU requested. The {GPU_RUN_MINUTES}-minute window starts when it is online.",
                   gpu_upgrades_remaining=remaining)


@app.get("/api/voices")
def voices():
    try:
        def go(client, found, meta):
            infer_meta = meta[found["infer"]]
            tts_meta = meta[found["tts"]] if found["tts"] else None
            out = {
                "f0_methods": infer_meta["choices"][I_INFER["f0"]] or ["rmvpe"],
                "export_formats": infer_meta["choices"][I_INFER["fmt"]] or ["WAV"],
                "model": VOICE_MODEL,
                "index": VOICE_INDEX,
                "tts_voices": (tts_meta["choices"][I_TTS["voice"]] or []) if tts_meta else [],
                "tts_available": bool(tts_meta),
            }
            return out
        return jsonify(ok=True, **with_reconnect(go))
    except ServerOffline as e:
        return fail(e, 503)
    except Exception as e:
        return fail(e, 502)


@app.post("/api/convert")
def convert():
    f = request.files.get("audio")
    if not f:
        return fail("Attach an audio file as 'audio'.")
    try:
        s = parse_settings(json.loads(request.form.get("settings", "{}")))
    except (ValueError, TypeError):
        return fail("Bad settings.")

    snap = get_snapshot()
    if not (snap and snap.get("link")):
        return fail("The voice server is still starting. Please try again in a moment.", 503)

    ext = Path(f.filename or "").suffix.lower()
    ext = ext if re.fullmatch(r"\.[a-z0-9]{2,5}", ext) else ".wav"
    src = WORK / f"in_{secrets.token_hex(4)}{ext}"
    f.save(src)

    def go(client, found, meta):
        args = list(meta[found["infer"]]["defaults"])
        up = client.predict(handle_file(str(src)), api_name=found["save"])
        args[I_INFER["terms"]] = True
        args[I_INFER["audio"]] = as_value(up[0])
        args[I_INFER["output"]] = as_value(up[1])
        args[I_INFER["pitch"]] = s["pitch"]
        args[I_INFER["index_rate"]] = s["index_rate"]
        for k in ("volume", "protect", "clean_strength", "f0", "fmt", "split", "autotune", "clean"):
            if k in s:
                args[I_INFER[k]] = s[k]
        args[I_INFER["model"]] = VOICE_MODEL
        args[I_INFER["index"]] = VOICE_INDEX
        return run_job(client, args, found["infer"])

    if not acquire_converter():
        src.unlink(missing_ok=True)
        return fail("The converter is busy. Please try again shortly.", 503)
    try:
        res = with_reconnect(go)
    except ServerOffline as e:
        return fail(e, 503)
    except Exception as e:
        request_recheck()
        return fail(f"Conversion failed: {e}", 502)
    finally:
        CONVERT_LOCK.release()
        src.unlink(missing_ok=True)
    return _store_result(res)


@app.post("/api/tts")
def tts():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return fail("Send some 'text' to speak.")
    try:
        s = parse_settings(body.get("settings", {}))
    except (ValueError, TypeError):
        return fail("Bad settings.")

    def go(client, found, meta):
        if not found["tts"]:
            raise RuntimeError("This Applio session has no text to speech tab.")
        args = list(meta[found["tts"]]["defaults"])
        args[I_TTS["terms"]] = True
        args[I_TTS["text"]] = text
        if "voice" in s:
            args[I_TTS["voice"]] = s["voice"]
        if "rate" in s:
            args[I_TTS["rate"]] = s["rate"]
        args[I_TTS["pitch"]] = s["pitch"]
        args[I_TTS["index_rate"]] = s["index_rate"]
        for k in ("protect", "f0", "fmt"):
            if k in s:
                args[I_TTS[k]] = s[k]
        args[I_TTS["model"]] = VOICE_MODEL
        args[I_TTS["index"]] = VOICE_INDEX
        return run_job(client, args, found["tts"])

    snap = get_snapshot()
    if not (snap and snap.get("link")):
        return fail("The voice server is still starting. Please try again in a moment.", 503)
    if not acquire_converter():
        return fail("The converter is busy. Please try again shortly.", 503)
    try:
        res = with_reconnect(go)
    except ServerOffline as e:
        return fail(e, 503)
    except Exception as e:
        request_recheck()
        return fail(f"Text to speech failed: {e}", 502)
    finally:
        CONVERT_LOCK.release()
    return _store_result(res)


@app.get("/api/result/<rid>")
def result(rid):
    with RESULTS_LOCK:
        entry = RESULTS.get(rid)
    if not entry or not entry[0].exists():
        return fail("Result not found (it may have expired or the server restarted).", 404)
    p = entry[0]
    mime = mimetypes.guess_type(str(p))[0] or "audio/wav"
    return send_file(p, mimetype=mime, as_attachment=bool(request.args.get("dl")),
                     download_name=f"converted_{rid}{p.suffix}")


# ---------- boot ----------

_STARTED = {"done": False}


def start_background():
    if _STARTED["done"]:
        return
    _STARTED["done"] = True
    threading.Thread(target=supervisor, daemon=True, name="supervisor").start()


if os.environ.get("RVC_NO_THREADS") != "1":
    start_background()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
