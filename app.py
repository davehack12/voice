"""
app.py: the whole voice changer backend, meant to run on Render (or anywhere).

Required environment variables:
    KAGGLE_USERNAME       your Kaggle username (from kaggle.json)
    KAGGLE_KEY            your legacy Kaggle API key (from kaggle.json)
    NGROK_AUTHTOKEN_1     first ngrok account authtoken
    NGROK_AUTHTOKEN_2     second ngrok account authtoken
    RVC_NTFY_TOPIC_A      CPU slot 1 ntfy topic (long random string).
    RVC_NTFY_TOPIC_B      CPU slot 2 ntfy topic (long random string).
    RVC_NTFY_TOPIC_GPU    GPU slot ntfy topic (long random string).

Optional:
    KERNEL_SLUG           default: rvc-gpu-server
    CACHE_SLUG            default: rvc-cache
    DATASET               default: supporttopal/sweet-female-rvc
    MODEL_NAME            default: sweet_female
    VOICE_MODEL           default: logs/sweet_female/sweet_female.pth
    VOICE_INDEX           default: logs/sweet_female/sweet_female.index
    ALLOWED_ORIGIN        default: *
    CPU_RUN_MINUTES       default: 600
    CPU_SPARE_PREP_MIN    default: 570
    CPU_KILL_OLD_MIN      default: 600
    GPU_RUN_MINUTES       default: 40
    GPU_MAX_EXTENDED_MIN  default: 90
    GPU_DAILY_RUNS        default: 6
    START_ON_BOOT         default: 0
    STATUS_TTL_SECONDS    default: 8
    NTFY_HISTORY_SECONDS  default: 43200
"""
import glob
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from gradio_client import Client, handle_file

# ---------- configuration ----------

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "supporttopal")
KERNEL_SLUG = os.environ.get("KERNEL_SLUG", "rvc-gpu-server")
CACHE_SLUG = os.environ.get("CACHE_SLUG", "rvc-cache")
DATASET = os.environ.get("DATASET", "supporttopal/sweet-female-rvc")
MODEL_NAME = os.environ.get("MODEL_NAME", "sweet_female")
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
NTFY = "https://ntfy.sh"

VOICE_MODEL = os.environ.get("VOICE_MODEL", "logs/sweet_female/sweet_female.pth")
VOICE_INDEX = os.environ.get("VOICE_INDEX", "logs/sweet_female/sweet_female.index")

TOPIC_A = os.environ.get("RVC_NTFY_TOPIC_A")
TOPIC_B = os.environ.get("RVC_NTFY_TOPIC_B")
TOPIC_GPU = os.environ.get("RVC_NTFY_TOPIC_GPU")

NGROK_TOKEN_1 = os.environ.get("NGROK_AUTHTOKEN_1")
NGROK_TOKEN_2 = os.environ.get("NGROK_AUTHTOKEN_2")

KAGGLE_KEY = os.environ.get("KAGGLE_KEY")

CPU_RUN_MINUTES      = int(os.environ.get("CPU_RUN_MINUTES", "600"))
CPU_SPARE_PREP_MIN   = int(os.environ.get("CPU_SPARE_PREP_MIN", "570"))
CPU_KILL_OLD_MIN     = int(os.environ.get("CPU_KILL_OLD_MIN", "600"))
GPU_RUN_MINUTES      = int(os.environ.get("GPU_RUN_MINUTES", "40"))
GPU_MAX_EXTENDED_MIN = int(os.environ.get("GPU_MAX_EXTENDED_MIN", "90"))
GPU_DAILY_RUNS       = int(os.environ.get("GPU_DAILY_RUNS", "6"))
START_ON_BOOT        = os.environ.get("START_ON_BOOT", "0") == "1"
STATUS_TTL_SECONDS   = int(os.environ.get("STATUS_TTL_SECONDS", "8"))
NTFY_HISTORY_SECONDS = int(os.environ.get("NTFY_HISTORY_SECONDS", "43200"))

KERNEL_ID = f"{KAGGLE_USERNAME}/{KERNEL_SLUG}"
CACHE_ID = f"{KAGGLE_USERNAME}/{CACHE_SLUG}"
DONE = {"COMPLETE", "ERROR", "CANCELACKNOWLEDGED"}
WORK = Path(tempfile.mkdtemp(prefix="rvc_backend_"))
BUILD = WORK / "kernel_build"

print(f"[config] KAGGLE_USERNAME={KAGGLE_USERNAME!r} KERNEL_SLUG={KERNEL_SLUG!r} "
      f"KERNEL_ID={KERNEL_ID!r} KEY_SET={bool(KAGGLE_KEY)} "
      f"NGROK1_SET={bool(NGROK_TOKEN_1)} NGROK2_SET={bool(NGROK_TOKEN_2)}",
      flush=True)

if not KAGGLE_KEY:
    print("[config] FATAL: KAGGLE_KEY is not set.", flush=True)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGIN}})

CONVERT_LOCK = threading.Lock()
RESULTS = {}
CLIENT_CACHE = {"url": None, "client": None, "endpoints": None}

# ---------- runtime state ----------

SLOTS_LOCK = threading.Lock()

SLOTS = [
    {"id": "cpu_a", "kind": "cpu", "topic": TOPIC_A, "pushed_at": 0,
     "push_lock": threading.Lock(), "stop_sent": False, "token": None,
     "hard_killed_at": 0},
    {"id": "cpu_b", "kind": "cpu", "topic": TOPIC_B, "pushed_at": 0,
     "push_lock": threading.Lock(), "stop_sent": False, "token": None,
     "hard_killed_at": 0},
    {"id": "gpu",   "kind": "gpu", "topic": TOPIC_GPU, "pushed_at": 0,
     "push_lock": threading.Lock(), "stop_sent": False, "token": None,
     "hard_killed_at": 0},
]
SLOT_BY_ID = {s["id"]: s for s in SLOTS}

UPGRADE_LOG = []
UPGRADE_LOG_LOCK = threading.Lock()

STATUS_CACHE = {"at": 0.0, "value": None}
STATUS_CACHE_LOCK = threading.Lock()

GPU_HOLD_UNTIL = {"at": 0}

WATCHER_FAILS = {"count": 0, "last_log": 0}
WATCHER_MAX_BACKOFF = 300

# ---------- kernel template ----------

KERNEL_TEMPLATE = r"""
import glob, json, os, re, shutil, subprocess, sys, threading, time, urllib.request

TOPIC = "__TOPIC__"
MAX_SECONDS = __MINUTES__ * 60
USE_CACHE = __USE_CACHE__
MODEL_NAME = "__MODEL_NAME__"
NGROK_TOKEN = "__NGROK__"
APPLIO = "/kaggle/working/Applio"
NTFY = "https://ntfy.sh"
started = time.time()
ENV = dict(os.environ, PYTHONUNBUFFERED="1")
proc = None


def say(msg):
    try:
        req = urllib.request.Request(f"{NTFY}/{TOPIC}", data=msg.encode(), method="POST")
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as e:
        print("ntfy post failed:", e, flush=True)


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
        time.sleep(15)
        tick += 1
        if time.time() - started > MAX_SECONDS:
            kill_everything("time limit reached, shutting down")
        if stop_requested():
            kill_everything("stop received, shutting down")
        if tick % 4 == 0:
            say(f"STATUS alive {int((time.time() - started) / 60)} min")


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

    time.sleep(30)

    say("STATUS opening ngrok tunnel")
    from pyngrok import conf, ngrok

    conf.get_default().auth_token = NGROK_TOKEN
    conf.get_default().monitor_thread = False

    existing = ngrok.get_tunnels(conf.get_default())
    if existing:
        public_url = existing[0].public_url
    else:
        public_url = ngrok.connect(7860, bind_tls=True).public_url

    say("LINK " + public_url)

    def reader():
        for line in proc.stdout:
            print(line, end="", flush=True)

    threading.Thread(target=reader, daemon=True).start()

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


def fail(message, code=400):
    return jsonify(ok=False, error=str(message)), code


# ---------- ntfy + kaggle helpers ----------

def ntfy_read(topic):
    since_ts = int(time.time() - NTFY_HISTORY_SECONDS)
    url = f"{NTFY}/{topic}/json?poll=1&since={since_ts}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:
        return []
    out = []
    for line in text.splitlines():
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("event") == "message":
            out.append(m)
    return out


def ntfy_send(topic, text):
    req = urllib.request.Request(f"{NTFY}/{topic}", data=text.encode(), method="POST")
    urllib.request.urlopen(req, timeout=20).read()


def find_link(msgs):
    link = None
    for m in msgs:
        msg = m.get("message", "")
        if msg == "STATUS finished":
            link = None
        elif msg.startswith("LINK "):
            link = msg[5:].strip()
    return link


def last_message(msgs):
    return msgs[-1] if msgs else None


def kaggle_cli(*args, timeout=120):
    """Run a kaggle CLI command with a hard timeout that kills the whole
    process group. subprocess.run with start_new_session=True and a single
    merged pipe is what makes the timeout actually fire; Popen.communicate
    can hang forever if a grandchild holds the stdout pipe.
    """
    env = dict(os.environ)
    try:
        p = subprocess.run(
            ["kaggle", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=timeout,
            start_new_session=True,
        )
        return p.returncode, (p.stdout or "").strip()
    except subprocess.TimeoutExpired:
        return 1, f"kaggle {' '.join(args)} timed out after {timeout}s"
    except FileNotFoundError as e:
        return 1, f"kaggle CLI not found: {e}"
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


def parse_state(out):
    m = re.search(r'status\s+"?([\w.]+)"?', out)
    if not m:
        return None
    return m.group(1).split(".")[-1].replace("_", "").upper()


def run_state():
    if not shutil.which("kaggle"):
        return None
    code, out = kaggle_cli("kernels", "status", KERNEL_ID, timeout=30)
    if code != 0:
        return None
    return parse_state(out)


def is_active(s):
    return s is not None and s not in DONE


def write_kernel(code, gpu):
    BUILD.mkdir(exist_ok=True)
    (BUILD / "script.py").write_text(code)
    meta = {
        "id": KERNEL_ID,
        "title": KERNEL_SLUG.replace("-", " "),
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if gpu else "false",
        "enable_internet": "true",
        "dataset_sources": [DATASET],
        "competition_sources": [],
        "kernel_sources": [CACHE_ID],
    }
    (BUILD / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


# ---------- token management ----------

TOKEN_1 = "1"
TOKEN_2 = "2"

def token_value(name):
    return NGROK_TOKEN_1 if name == TOKEN_1 else NGROK_TOKEN_2

def claim_token(slot, preferred=None):
    with SLOTS_LOCK:
        held = {s["token"] for s in SLOTS if s is not slot and s["token"]}
        if preferred and preferred not in held:
            slot["token"] = preferred
            return preferred
        for t in (TOKEN_1, TOKEN_2):
            if t not in held:
                slot["token"] = t
                return t
    return None


# ---------- slot bookkeeping ----------

def build_kernel_code(topic, minutes, token_name):
    return (KERNEL_TEMPLATE
            .replace("__TOPIC__", topic)
            .replace("__MINUTES__", str(minutes))
            .replace("__USE_CACHE__", "True")
            .replace("__MODEL_NAME__", MODEL_NAME)
            .replace("__NGROK__", token_value(token_name) or ""))


def push_slot(slot, minutes, gpu, preferred_token=None):
    print(f"[push] entering push_slot for {slot['id']}", flush=True)

    if not NGROK_TOKEN_1 or not NGROK_TOKEN_2:
        print("[push] rejected: missing ngrok tokens", flush=True)
        return False, {"error": "Both NGROK_AUTHTOKEN_1 and NGROK_AUTHTOKEN_2 must be set."}
    if not KAGGLE_KEY:
        print("[push] rejected: missing KAGGLE_KEY", flush=True)
        return False, {"error": "KAGGLE_KEY is not set on the server."}
    if not slot["topic"]:
        print(f"[push] rejected: missing topic for {slot['id']}", flush=True)
        return False, {"error": f"Topic for slot {slot['id']} is not set."}

    with SLOTS_LOCK:
        with slot["push_lock"]:
            if slot["pushed_at"]:
                print(f"[push] rejected: {slot['id']} already pushed", flush=True)
                return False, {"error": "This slot was already pushed recently."}

            token = claim_token(slot, preferred=preferred_token)
            if not token:
                print("[push] rejected: no ngrok token free", flush=True)
                return False, {"error": "No ngrok token free. Both slots are holding one."}

            code = build_kernel_code(slot["topic"], minutes, token)
            write_kernel(code, gpu=gpu)

            print(f"[push] submitting {slot['id']} on token {token}", flush=True)
            rc, out = kaggle_cli(
                "kernels", "push", "-p", str(BUILD),
                "-t", str(minutes * 60 + 300),
                timeout=120,
            )
            if rc != 0:
                slot["token"] = None
                print(f"[push] failed for {slot['id']}: {out}", flush=True)
                return False, {"error": f"Kaggle push failed: {out}"}

            slot["pushed_at"] = int(time.time())
            slot["stop_sent"] = False
            slot["hard_killed_at"] = 0
            print(f"[push] ok for {slot['id']} on token {token}: {out[:200]}", flush=True)
            return True, {"kaggle_output": out, "token": token}


def slot_link(slot):
    if not slot["topic"]:
        return None
    msgs = ntfy_read(slot["topic"])
    return find_link(msgs)


def slot_last_message(slot):
    if not slot["topic"]:
        return None
    msgs = ntfy_read(slot["topic"])
    last = last_message(msgs)
    return last["message"] if last else None


def slot_state_dict(slot):
    link = slot_link(slot)
    pushed = slot["pushed_at"]
    age = int(time.time() - pushed) if pushed else None
    return {
        "id": slot["id"],
        "kind": slot["kind"],
        "link": link,
        "pushed_at": pushed or None,
        "age_seconds": age,
        "last_message": slot_last_message(slot),
        "token": slot["token"],
        "hard_killed_at": slot["hard_killed_at"] or None,
    }


def record_gpu_upgrade():
    now = time.time()
    with UPGRADE_LOG_LOCK:
        cutoff = now - 24 * 3600
        UPGRADE_LOG[:] = [t for t in UPGRADE_LOG if t > cutoff]
        if len(UPGRADE_LOG) >= GPU_DAILY_RUNS:
            return False, GPU_DAILY_RUNS - len(UPGRADE_LOG)
        UPGRADE_LOG.append(now)
        return True, GPU_DAILY_RUNS - len(UPGRADE_LOG)


def upgrades_remaining():
    now = time.time()
    with UPGRADE_LOG_LOCK:
        cutoff = now - 24 * 3600
        UPGRADE_LOG[:] = [t for t in UPGRADE_LOG if t > cutoff]
        return GPU_DAILY_RUNS - len(UPGRADE_LOG)


# ---------- hard kill ----------

KILL_VERIFY_SECONDS = 90

def hard_kill_slot(slot):
    with SLOTS_LOCK:
        if slot["stop_sent"] and slot["pushed_at"] == 0:
            return
        slot["stop_sent"] = True
        slot["pushed_at"] = 0
        slot["hard_killed_at"] = int(time.time())
        try:
            ntfy_send(slot["topic"], "STOP")
        except Exception:
            pass

        def second():
            time.sleep(4)
            try:
                ntfy_send(slot["topic"], "STOP")
            except Exception:
                pass
        threading.Thread(target=second, daemon=True).start()


def confirm_dead(slot, timeout=KILL_VERIFY_SECONDS):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if slot_link(slot) is None:
            return True
        time.sleep(3)
    return False


# ---------- current status ----------

def _pick_cpu_primary(state_a, state_b):
    live = [s for s in (state_a, state_b) if s["link"]]
    pool = live or [state_a, state_b]
    return max(pool, key=lambda s: s["pushed_at"] or 0)


def current_status():
    cpu_a = SLOT_BY_ID["cpu_a"]
    cpu_b = SLOT_BY_ID["cpu_b"]
    gpu   = SLOT_BY_ID["gpu"]

    state_a = slot_state_dict(cpu_a)
    state_b = slot_state_dict(cpu_b)
    state_g = slot_state_dict(gpu)

    cpu_primary = _pick_cpu_primary(state_a, state_b)
    cpu_spare = state_b if cpu_primary["id"] == "cpu_a" else state_a

    gpu_live = bool(state_g["link"])
    if gpu_live:
        primary = "gpu"
        link = state_g["link"]
    else:
        primary = "cpu"
        link = cpu_primary["link"]

    kaggle = run_state()

    if cpu_spare["link"]:
        cpu_spare_state = "ready"
    elif cpu_spare["pushed_at"]:
        cpu_spare_state = "booting"
    else:
        cpu_spare_state = "idle"

    if gpu_live:
        gpu_state = "ready"
    elif state_g["pushed_at"]:
        gpu_state = "booting"
    else:
        gpu_state = "idle"

    if link:
        server_state = "online"
    elif cpu_primary["pushed_at"] or cpu_spare["pushed_at"] or state_g["pushed_at"]:
        server_state = "booting"
    else:
        server_state = "idle"

    last_msg = state_g["last_message"] if primary == "gpu" else cpu_primary["last_message"]

    return {
        "kaggle_state": kaggle,
        "active": is_active(kaggle) or bool(link) or server_state == "booting",
        "server_state": server_state,
        "link": link,
        "primary": primary,
        "cpu_primary_age_min": (
            int(cpu_primary["age_seconds"] / 60) if cpu_primary["age_seconds"] else None
        ),
        "cpu_spare_state": cpu_spare_state,
        "gpu_state": gpu_state,
        "gpu_upgrades_remaining": upgrades_remaining(),
        "gpu_upgrades_per_day": GPU_DAILY_RUNS,
        "last_message": last_msg,
        "last_message_age_seconds": None,
        "cpu_a_token": state_a["token"],
        "cpu_b_token": state_b["token"],
        "gpu_token": state_g["token"],
    }


def cached_status():
    now = time.time()
    with STATUS_CACHE_LOCK:
        value = STATUS_CACHE["value"]
        at = STATUS_CACHE["at"]
        if value is not None and (now - at) < STATUS_TTL_SECONDS:
            return value
    fresh = current_status()
    with STATUS_CACHE_LOCK:
        STATUS_CACHE["at"] = time.time()
        STATUS_CACHE["value"] = fresh
    return fresh


# ---------- background threads ----------

def _cpu_states():
    return (slot_state_dict(SLOT_BY_ID["cpu_a"]),
            slot_state_dict(SLOT_BY_ID["cpu_b"]))


def cpu_rotator():
    while True:
        try:
            if not (TOPIC_A and TOPIC_B):
                time.sleep(60)
                continue

            cpu_a = SLOT_BY_ID["cpu_a"]
            cpu_b = SLOT_BY_ID["cpu_b"]
            state_a, state_b = _cpu_states()
            cpu_primary = _pick_cpu_primary(state_a, state_b)

            spare_id = "cpu_b" if cpu_primary["id"] == "cpu_a" else "cpu_a"
            spare = SLOT_BY_ID[spare_id]
            spare_state = state_b if spare_id == "cpu_b" else state_a
            primary_slot = SLOT_BY_ID[cpu_primary["id"]]

            primary_age_min = (
                int((cpu_primary["age_seconds"] or 0) / 60)
                if cpu_primary["age_seconds"] else 0
            )

            gpu = SLOT_BY_ID["gpu"]
            gpu_state = slot_state_dict(gpu)
            gpu_active = bool(gpu_state["link"]) or bool(gpu["pushed_at"])

            if (not cpu_primary["link"]
                    and not cpu_primary["pushed_at"]
                    and not spare_state["pushed_at"]
                    and not spare_state["link"]):
                push_slot(primary_slot, CPU_RUN_MINUTES, gpu=False,
                          preferred_token=TOKEN_1)
                time.sleep(60)
                continue

            if spare_state["link"]:
                if not primary_slot["stop_sent"]:
                    hard_kill_slot(primary_slot)
                time.sleep(60)
                continue

            if (primary_age_min >= CPU_SPARE_PREP_MIN
                    and primary_age_min < CPU_KILL_OLD_MIN
                    and not spare_state["pushed_at"]
                    and not spare_state["link"]):

                primary_token = cpu_primary.get("token")
                preferred = TOKEN_2 if primary_token == TOKEN_1 else TOKEN_1

                if gpu_active and gpu.get("token") == preferred:
                    preferred = primary_token

                ok, _ = push_slot(spare, CPU_RUN_MINUTES, gpu=False,
                                  preferred_token=preferred)
                if not ok:
                    hard_kill_slot(primary_slot)
                    time.sleep(6)
                    freed = primary_slot["token"]
                    primary_slot["token"] = None
                    push_slot(spare, CPU_RUN_MINUTES, gpu=False,
                              preferred_token=freed)
                time.sleep(60)
                continue

            if primary_age_min >= CPU_KILL_OLD_MIN and not primary_slot["stop_sent"]:
                hard_kill_slot(primary_slot)
                time.sleep(60)
                continue

        except Exception as e:
            print(f"[rotator] exception: {e}", flush=True)
        time.sleep(60)


def gpu_watchdog():
    while True:
        try:
            gpu = SLOT_BY_ID["gpu"]
            pushed = gpu["pushed_at"]
            if pushed and TOPIC_GPU:
                elapsed_min = int((time.time() - pushed) / 60)

                cpu_link_live = bool(
                    slot_link(SLOT_BY_ID["cpu_a"]) or slot_link(SLOT_BY_ID["cpu_b"])
                )

                if elapsed_min >= GPU_RUN_MINUTES:
                    if cpu_link_live:
                        hard_kill_slot(gpu)
                    else:
                        if elapsed_min >= GPU_MAX_EXTENDED_MIN:
                            hard_kill_slot(gpu)
                        else:
                            if not GPU_HOLD_UNTIL["at"]:
                                GPU_HOLD_UNTIL["at"] = int(time.time())
                else:
                    GPU_HOLD_UNTIL["at"] = 0
        except Exception as e:
            print(f"[gpu-watchdog] exception: {e}", flush=True)
        time.sleep(60)


def link_watcher():
    while True:
        try:
            state_a, state_b = _cpu_states()
            gpu = slot_state_dict(SLOT_BY_ID["gpu"])

            any_link = bool(state_a["link"] or state_b["link"] or gpu["link"])
            any_push = bool(
                state_a["pushed_at"] or state_b["pushed_at"] or gpu["pushed_at"]
            )

            print(f"[watcher] tick link={any_link} push={any_push}", flush=True)

            if not any_link and not any_push:
                cpu_primary = _pick_cpu_primary(state_a, state_b)
                print(f"[watcher] attempting push to {cpu_primary['id']}", flush=True)
                ok, info = push_slot(SLOT_BY_ID[cpu_primary["id"]],
                                     CPU_RUN_MINUTES,
                                     gpu=False, preferred_token=TOKEN_1)
                if ok:
                    WATCHER_FAILS["count"] = 0
                    print(f"[watcher] push ok: {info}", flush=True)
                else:
                    WATCHER_FAILS["count"] += 1
                    now = time.time()
                    if now - WATCHER_FAILS["last_log"] > 30:
                        WATCHER_FAILS["last_log"] = now
                        print(f"[watcher] push failed ({WATCHER_FAILS['count']} in a row): {info.get('error')}", flush=True)
                    backoff = min(WATCHER_MAX_BACKOFF, WATCHER_FAILS["count"] * 30)
                    time.sleep(backoff)
                    continue
        except Exception as e:
            print(f"[watcher] exception: {e}", flush=True)
        time.sleep(30)


# ---------- Applio proxy ----------

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
    st = cached_status()
    link = st["link"]
    if not link:
        raise RuntimeError("No server is online. Start it first.")
    if CLIENT_CACHE["url"] == link and CLIENT_CACHE["client"] is not None:
        return CLIENT_CACHE["client"], CLIENT_CACHE["endpoints"], link
    client = Client(link, verbose=False)
    found, meta = discover(client)
    CLIENT_CACHE.update(url=link, client=client, endpoints=(found, meta))
    return client, (found, meta), link


def with_reconnect(fn):
    try:
        client, (found, meta), link = get_endpoints()
        return fn(client, found, meta)
    except Exception as first:
        CLIENT_CACHE.update(url=None, client=None, endpoints=None)
        try:
            client, (found, meta), link = get_endpoints()
            return fn(client, found, meta)
        except Exception:
            raise first


# ---------- routes ----------

@app.get("/")
def home():
    return jsonify(ok=True, service="voice changer backend")


@app.get("/api/status")
def status():
    if not TOPIC_A or not TOPIC_B or not TOPIC_GPU:
        return fail("One or more RVC_NTFY_TOPIC_* variables are missing.", 500)
    return jsonify(ok=True, **cached_status())


@app.post("/api/start")
def start():
    if not TOPIC_A or not TOPIC_B or not TOPIC_GPU:
        return fail("One or more RVC_NTFY_TOPIC_* variables are missing.", 500)

    cpu_a = SLOT_BY_ID["cpu_a"]
    cpu_b = SLOT_BY_ID["cpu_b"]
    state_a, state_b = _cpu_states()

    if (state_a["link"] or state_b["link"]
            or cpu_a["pushed_at"] or cpu_b["pushed_at"]):
        return jsonify(ok=True, already_running=True, **cached_status())

    primary = _pick_cpu_primary(state_a, state_b)
    ok, info = push_slot(SLOT_BY_ID[primary["id"]], CPU_RUN_MINUTES,
                         gpu=False, preferred_token=TOKEN_1)
    if not ok:
        return fail(info.get("error", "Push failed"), 500)
    return jsonify(ok=True, already_running=False,
                   message="CPU push submitted. Poll /api/status for the link.",
                   kaggle_output=info.get("kaggle_output"))


@app.post("/api/stop")
def stop():
    if not TOPIC_A or not TOPIC_B or not TOPIC_GPU:
        return fail("One or more RVC_NTFY_TOPIC_* variables are missing.", 500)

    for slot in SLOTS:
        hard_kill_slot(slot)
    CLIENT_CACHE.update(url=None, client=None, endpoints=None)
    with STATUS_CACHE_LOCK:
        STATUS_CACHE["value"] = None
        STATUS_CACHE["at"] = 0.0
    return jsonify(ok=True, message="Stop sent to all slots.")


@app.post("/api/upgrade-gpu")
def upgrade_gpu():
    if not TOPIC_GPU:
        return fail("RVC_NTFY_TOPIC_GPU is not set on the server.", 500)

    gpu = SLOT_BY_ID["gpu"]
    state_g = slot_state_dict(gpu)

    if state_g["link"] or gpu["pushed_at"]:
        return jsonify(ok=True, already_running=True,
                       message="GPU is already running or booting.",
                       **cached_status())

    allowed, remaining = record_gpu_upgrade()
    if not allowed:
        return fail(f"Daily GPU limit reached. Try again later. Remaining: {remaining}.", 429)

    cpu_primary = _pick_cpu_primary(*_cpu_states())
    primary_token = cpu_primary.get("token")
    preferred = TOKEN_2 if primary_token == TOKEN_1 else TOKEN_1

    ok, info = push_slot(gpu, GPU_RUN_MINUTES, gpu=True, preferred_token=preferred)
    if not ok:
        return fail(info.get("error", "GPU push failed"), 500)

    GPU_HOLD_UNTIL["at"] = 0

    with STATUS_CACHE_LOCK:
        STATUS_CACHE["value"] = None
        STATUS_CACHE["at"] = 0.0

    return jsonify(ok=True, already_running=False,
                   message="GPU push submitted. The 40-minute window starts now.",
                   gpu_upgrades_remaining=remaining,
                   kaggle_output=info.get("kaggle_output"))


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
            }
            if tts_meta:
                out["tts_voices"] = tts_meta["choices"][I_TTS["voice"]] or []
                out["tts_available"] = True
            else:
                out["tts_voices"] = []
                out["tts_available"] = False
            return out
        return jsonify(ok=True, **with_reconnect(go))
    except Exception as e:
        return fail(e, 502)


@app.post("/api/convert")
def convert():
    f = request.files.get("audio")
    if not f:
        return fail("Attach an audio file as 'audio'.")
    try:
        cfg = json.loads(request.form.get("settings", "{}"))
    except ValueError:
        return fail("Bad settings JSON.")

    ext = Path(f.filename or "").suffix.lower()
    ext = ext if re.fullmatch(r"\.[a-z0-9]{2,5}", ext) else ".wav"
    src = WORK / f"in_{secrets.token_hex(4)}{ext}"
    f.save(src)

    def go(client, found, meta):
        args = list(meta[found["infer"]]["defaults"])
        with CONVERT_LOCK:
            up = client.predict(handle_file(str(src)), api_name=found["save"])
            args[I_INFER["terms"]] = True
            args[I_INFER["audio"]] = as_value(up[0])
            args[I_INFER["output"]] = as_value(up[1])
            args[I_INFER["pitch"]] = int(cfg.get("pitch", 10))
            args[I_INFER["index_rate"]] = float(cfg.get("index_rate", 0.3))
            if "volume" in cfg:
                args[I_INFER["volume"]] = float(cfg["volume"])
            if "protect" in cfg:
                args[I_INFER["protect"]] = float(cfg["protect"])
            if cfg.get("f0"):
                args[I_INFER["f0"]] = cfg["f0"]
            if cfg.get("fmt"):
                args[I_INFER["fmt"]] = cfg["fmt"]
            for key in ("split", "autotune", "clean"):
                if key in cfg:
                    args[I_INFER[key]] = bool(cfg[key])
            if "clean_strength" in cfg:
                args[I_INFER["clean_strength"]] = float(cfg["clean_strength"])
            args[I_INFER["model"]] = VOICE_MODEL
            args[I_INFER["index"]] = VOICE_INDEX
            return client.predict(*args, api_name=found["infer"])

    try:
        res = with_reconnect(go)
    except Exception as e:
        return fail(f"Conversion failed: {e}", 502)
    finally:
        src.unlink(missing_ok=True)
    return _store_result(res)


@app.post("/api/tts")
def tts():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return fail("Send some 'text' to speak.")
    cfg = body.get("settings", {})

    def go(client, found, meta):
        if not found["tts"]:
            raise RuntimeError("This Applio session has no text to speech tab.")
        args = list(meta[found["tts"]]["defaults"])
        with CONVERT_LOCK:
            args[I_TTS["terms"]] = True
            args[I_TTS["text"]] = text
            if cfg.get("voice"):
                args[I_TTS["voice"]] = cfg["voice"]
            if "rate" in cfg:
                args[I_TTS["rate"]] = int(cfg["rate"])
            args[I_TTS["pitch"]] = int(cfg.get("pitch", 10))
            args[I_TTS["index_rate"]] = float(cfg.get("index_rate", 0.3))
            if "protect" in cfg:
                args[I_TTS["protect"]] = float(cfg["protect"])
            if cfg.get("f0"):
                args[I_TTS["f0"]] = cfg["f0"]
            if cfg.get("fmt"):
                args[I_TTS["fmt"]] = cfg["fmt"]
            args[I_TTS["model"]] = VOICE_MODEL
            args[I_TTS["index"]] = VOICE_INDEX
            return client.predict(*args, api_name=found["tts"])

    try:
        res = with_reconnect(go)
    except Exception as e:
        return fail(f"Text to speech failed: {e}", 502)
    return _store_result(res)


def _store_result(res):
    message, out = res[0], as_value(res[1])
    if not out or not os.path.exists(str(out)):
        return fail(message or "Applio returned no audio.", 502)
    rid = secrets.token_hex(6)
    dest = WORK / f"out_{rid}{Path(str(out)).suffix or '.wav'}"
    shutil.copy(str(out), dest)
    RESULTS[rid] = dest
    return jsonify(ok=True, id=rid, message=message)


@app.get("/api/result/<rid>")
def result(rid):
    p = RESULTS.get(rid)
    if not p or not p.exists():
        return fail("Result not found (the server may have restarted since it was made).", 404)
    mime = mimetypes.guess_type(str(p))[0] or "audio/wav"
    return send_file(p, mimetype=mime, as_attachment=bool(request.args.get("dl")),
                     download_name=f"converted_{rid}{p.suffix}")


# ---------- boot ----------

def _boot():
    if START_ON_BOOT:
        try:
            time.sleep(5)
            state_a, state_b = _cpu_states()
            primary = _pick_cpu_primary(state_a, state_b)
            if not primary["link"] and not primary["pushed_at"]:
                push_slot(SLOT_BY_ID[primary["id"]], CPU_RUN_MINUTES,
                          gpu=False, preferred_token=TOKEN_1)
        except Exception:
            pass


threading.Thread(target=cpu_rotator, daemon=True).start()
threading.Thread(target=gpu_watchdog, daemon=True).start()
threading.Thread(target=link_watcher, daemon=True).start()
threading.Thread(target=_boot, daemon=True).start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
