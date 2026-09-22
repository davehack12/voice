"""
app.py: the whole voice changer backend, meant to run on Render (or anywhere).

Required environment variables (set these in Render's dashboard, never in code):
    KAGGLE_API_TOKEN   your Kaggle API token, from kaggle.com/settings/api
    RVC_NTFY_TOPIC     a long random string, e.g. output of:
                           python3 -c "import secrets; print(secrets.token_hex(20))"
                       This is the private channel the Kaggle run and this backend
                       use to talk to each other. Anyone who knows it can read your
                       link and stop your run, so keep it as secret as a password.

Optional environment variables (sensible defaults are used otherwise):
    KAGGLE_USERNAME    default: supporttopal
    KERNEL_SLUG        default: rvc-gpu-server
    CACHE_SLUG         default: rvc-cache
    DATASET            default: supporttopal/sweet-female-rvc
    MODEL_NAME         default: sweet_female
    DEFAULT_MINUTES    default: 60   (hard time limit for a run, in minutes)
    ALLOWED_ORIGIN     default: *    (set to your site's URL once you have one)

No accounts, database, or file storage are required. State is never kept only
in this process's memory, since Render's free tier restarts it after idling,
so every status check re-asks Kaggle and re-checks the ntfy channel directly.
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

from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
from gradio_client import Client, handle_file

# ---------- configuration ----------

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "supporttopal")
KERNEL_SLUG = os.environ.get("KERNEL_SLUG", "rvc-gpu-server")
CACHE_SLUG = os.environ.get("CACHE_SLUG", "rvc-cache")
DATASET = os.environ.get("DATASET", "supporttopal/sweet-female-rvc")
MODEL_NAME = os.environ.get("MODEL_NAME", "sweet_female")
DEFAULT_MINUTES = int(os.environ.get("DEFAULT_MINUTES", "60"))
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
NTFY = "https://ntfy.sh"

TOPIC = os.environ.get("RVC_NTFY_TOPIC")
KERNEL_ID = f"{KAGGLE_USERNAME}/{KERNEL_SLUG}"
CACHE_ID = f"{KAGGLE_USERNAME}/{CACHE_SLUG}"
DONE = {"COMPLETE", "ERROR", "CANCELACKNOWLEDGED"}
WORK = Path(tempfile.mkdtemp(prefix="rvc_backend_"))
BUILD = WORK / "kernel_build"

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGIN}})
CONVERT_LOCK = threading.Lock()
RESULTS = {}
CLIENT_CACHE = {"url": None, "client": None, "endpoints": None}

# The run script itself. Identical to the one tested locally: it listens for STOP
# on the ntfy topic from the moment it starts (not only once Applio is up), sends a
# heartbeat every minute, and shuts itself and every child process down on STOP or
# once the time limit is reached.
KERNEL_TEMPLATE = r"""
import glob, json, os, re, shutil, subprocess, sys, threading, time, urllib.request

TOPIC = "__TOPIC__"
MAX_SECONDS = __MINUTES__ * 60
USE_CACHE = __USE_CACHE__
MODEL_NAME = "__MODEL_NAME__"
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
    # SETUP START
    sh(f"git clone --depth 1 https://github.com/IAHispano/Applio.git {APPLIO}")
    sh("apt-get update -y")
    sh("apt-get install -y portaudio19-dev libportaudio2")
    if libs is None:
        sh("pip install -q uv")
        sh("uv pip install -q -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match --system", cwd=APPLIO)
    say("STATUS downloading Applio models")
    sh(f"{sys.executable} core.py prerequisites --models", cwd=APPLIO)
    # SETUP END

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
        [sys.executable, "-u", "app.py", "--listen", "--client"],
        cwd=APPLIO, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    time.sleep(30)

    say("STATUS opening ngrok tunnel")
    from kaggle_secrets import UserSecretsClient
    from pyngrok import conf, ngrok

    conf.get_default().auth_token = UserSecretsClient().get_secret("NGROK_AUTHTOKEN")
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
    url = f"{NTFY}/{topic}/json?poll=1&since=all"
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
            # A run ended here (normal exit, STOP, time limit, or crash) - any
            # link posted before this point belongs to that finished run, so
            # forget it rather than showing a dead link for the next run.
            link = None
        elif msg.startswith("LINK "):
            link = msg[5:].strip()
    return link


def kaggle_cli(*args):
    env = dict(os.environ)
    p = subprocess.run(["kaggle", *args], capture_output=True, text=True, env=env)
    return p.returncode, (p.stdout + p.stderr).strip()


def parse_state(out):
    m = re.search(r'status\s+"?([\w.]+)"?', out)
    if not m:
        return None
    return m.group(1).split(".")[-1].replace("_", "").upper()


def run_state():
    if not shutil.which("kaggle"):
        return None
    code, out = kaggle_cli("kernels", "status", KERNEL_ID)
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


def current_status():
    s = run_state()
    msgs = ntfy_read(TOPIC) if TOPIC else []
    link = find_link(msgs)
    last = msgs[-1] if msgs else None
    age = int(time.time() - last["time"]) if last else None
    return {
        "kaggle_state": s,
        "active": is_active(s),
        "link": link if is_active(s) else None,
        "last_message": last["message"] if last else None,
        "last_message_age_seconds": age,
    }


# ---------- Applio proxy (inference + text to speech) ----------

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
    """gradio_client reports a Dropdown's options as a `Literal['a', 'b']` type string."""
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
    """Reuses a live connection if we already have one; otherwise reconnects
    using whatever link is posted on the ntfy channel right now."""
    st = current_status()
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
    """Runs fn(client, found, meta) once; if it fails, drops the cached
    connection and retries once against a freshly discovered link."""
    try:
        client, (found, meta), link = get_endpoints()
        return fn(client, found, meta)
    except Exception:
        CLIENT_CACHE.update(url=None, client=None, endpoints=None)
        client, (found, meta), link = get_endpoints()
        return fn(client, found, meta)


# ---------- routes ----------

@app.get("/")
def home():
    return jsonify(ok=True, service="voice changer backend")


@app.get("/api/status")
def status():
    if not TOPIC:
        return fail("RVC_NTFY_TOPIC is not set on the server.", 500)
    return jsonify(ok=True, **current_status())


@app.post("/api/start")
def start():
    if not TOPIC:
        return fail("RVC_NTFY_TOPIC is not set on the server.", 500)
    if not shutil.which("kaggle"):
        return fail("The kaggle command is not available on this server.", 500)
    body = request.get_json(silent=True) or {}
    cpu = bool(body.get("cpu", False))
    minutes = int(body.get("minutes", DEFAULT_MINUTES))

    st = current_status()
    if st["active"]:
        return jsonify(ok=True, already_running=True, **st)

    code = (KERNEL_TEMPLATE
            .replace("__TOPIC__", TOPIC)
            .replace("__MINUTES__", str(minutes))
            .replace("__USE_CACHE__", "True")
            .replace("__MODEL_NAME__", MODEL_NAME))
    write_kernel(code, gpu=not cpu)
    rc, out = kaggle_cli("kernels", "push", "-p", str(BUILD), "-t", str(minutes * 60 + 300))
    if rc != 0:
        return fail(f"Kaggle push failed: {out}", 500)
    return jsonify(ok=True, already_running=False, message="Push submitted. Poll /api/status for the link.",
                   kaggle_output=out)


@app.post("/api/stop")
def stop():
    if not TOPIC:
        return fail("RVC_NTFY_TOPIC is not set on the server.", 500)
    st = current_status()
    if not st["active"]:
        return jsonify(ok=True, message=f"Nothing is running (Kaggle state: {st['kaggle_state']}).", **st)
    ntfy_send(TOPIC, "STOP")
    CLIENT_CACHE.update(url=None, client=None, endpoints=None)
    return jsonify(ok=True, message="Stop signal sent. Poll /api/status to see it end.")


@app.get("/api/voices")
def voices():
    try:
        def go(client, found, meta):
            infer_meta = meta[found["infer"]]
            tts_meta = meta[found["tts"]] if found["tts"] else None
            out = {
                "f0_methods": infer_meta["choices"][I_INFER["f0"]] or ["rmvpe"],
                "export_formats": infer_meta["choices"][I_INFER["fmt"]] or ["WAV"],
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
            if cfg.get("model"):
                args[I_INFER["model"]] = cfg["model"]
            if cfg.get("index"):
                args[I_INFER["index"]] = cfg["index"]
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
            if cfg.get("model"):
                args[I_TTS["model"]] = cfg["model"]
            if cfg.get("index"):
                args[I_TTS["index"]] = cfg["index"]
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
