#!/usr/bin/env python3
"""本地练习服务：在浏览器里写代码，用本机的 Python + PyTorch 跑测试。

    uv run serve.py              # 构建网页并打开 http://localhost:8000
    uv run serve.py --port 8001  # 换端口
    uv run serve.py --no-open    # 不自动打开浏览器
    uv run serve.py --check      # 自检：不起服务，只跑两道题确认运行链路正常（CI 用）

网页上点“运行测试”时，编辑器里的代码会发到这个服务，在子进程里跑这道题目录下的
test_*.py（和 `uv run run.py <题名>` 是同一份测试），超过 30 秒会被终止。

安全：只监听 127.0.0.1；只接受 Host / Origin 是本服务自己的请求，并要求自定义请求头，
其他网站不能借你的浏览器往这里提交代码。
"""
import argparse
import contextlib
import http.server
import importlib.metadata
import importlib.util
import inspect
import io
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parent
SITE = ROOT / "site" / "index.html"
EX_DIR = ROOT / "exercises"
TIMEOUT_S = 30
MAX_BODY = 1_000_000
SLUG_RE = re.compile(r"^[a-z0-9_]+$")
RESULT_MARK = "\x00RH_RESULT\x00"


# ---------------------------------------------------------------- 子进程里跑一道题
def find_exercise(slug):
    if not SLUG_RE.match(slug or ""):
        return None
    hits = [d for d in EX_DIR.glob(f"*/{slug}") if d.is_dir() and any(d.glob("test_*.py"))]
    return hits[0] if len(hits) == 1 else None


def _fmt(e, base):
    """只保留用户代码和测试文件里的帧，去掉 torch / 标准库内部的帧。"""
    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename.startswith(base)]
    head = ("Traceback (most recent call last):\n" + "".join(traceback.format_list(frames))) if frames else ""
    text = head + "".join(traceback.format_exception_only(type(e), e))
    text = text.replace(base + os.sep, "")
    lines = text.rstrip().splitlines()
    if len(lines) > 40:
        lines = lines[:12] + ["..."] + lines[-27:]
    return "\n".join(lines)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_job(slug, code):
    """在当前进程里跑：返回和网页 Pyodide 运行器相同格式的结果。"""
    ex = find_exercise(slug)
    if ex is None:
        return {"error": f"找不到题目 {slug}"}
    res = {"import_error": None, "tests": [], "stdout": ""}
    buf = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="rh_") as d:
        d = os.path.realpath(d)
        user = os.path.join(d, "template.py")
        pathlib.Path(user).write_text(code, encoding="utf-8")
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                mod = _load("rh_user", user)
            except BaseException as e:  # noqa: BLE001
                res["import_error"] = _fmt(e, d)
                mod = None
            if mod is not None:
                for tf in sorted(ex.glob("test_*.py")):
                    tp = os.path.join(d, tf.name)
                    pathlib.Path(tp).write_text(tf.read_text(encoding="utf-8"), encoding="utf-8")
                    try:
                        tm = _load("rh_test_" + tf.stem, tp)
                    except BaseException as e:  # noqa: BLE001
                        res["tests"].append({"name": tf.name, "status": "error", "detail": _fmt(e, d), "ms": 0})
                        continue
                    for name, fn in list(vars(tm).items()):
                        if not (name.startswith("test_") and callable(fn)):
                            continue
                        t0 = time.perf_counter()
                        try:
                            fn(mod) if len(inspect.signature(fn).parameters) >= 1 else fn()
                            st, detail = "pass", ""
                        except NotImplementedError:
                            st, detail = "todo", ""
                        except BaseException as e:  # noqa: BLE001
                            st, detail = "fail", _fmt(e, d)
                        res["tests"].append({"name": name, "status": st, "detail": detail,
                                             "ms": round((time.perf_counter() - t0) * 1000)})
    res["stdout"] = buf.getvalue()[-4000:]
    return res


def job_main():
    job = json.loads(sys.stdin.read())
    out = run_job(job.get("slug"), job.get("code", ""))
    sys.__stdout__.write(RESULT_MARK + json.dumps(out, ensure_ascii=False) + "\n")


def run_in_subprocess(slug, code):
    """每次运行开一个新进程：互不影响，死循环可以直接杀掉。"""
    try:
        p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), "--job"],
                           input=json.dumps({"slug": slug, "code": code}), capture_output=True,
                           text=True, encoding="utf-8", timeout=TIMEOUT_S, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    for line in reversed(p.stdout.splitlines()):
        if line.startswith(RESULT_MARK):
            return json.loads(line[len(RESULT_MARK):])
    tail = (p.stderr or p.stdout or "").strip().splitlines()[-20:]
    return {"error": "运行进程异常退出（退出码 %s）\n%s" % (p.returncode, "\n".join(tail))}


def versions():
    try:
        torch_v = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_v = None
    return {"ok": True, "python": platform.python_version(), "torch": torch_v}


# ---------------------------------------------------------------- HTTP 服务
class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "ReviewHelper"

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _trusted(self):
        host = (self.headers.get("Host") or "").lower()
        origin = self.headers.get("Origin")
        ok_hosts = {f"localhost:{self.server.server_port}", f"127.0.0.1:{self.server.server_port}"}
        return host in ok_hosts and (origin is None or origin.lower() in {"http://" + h for h in ok_hosts})

    def do_GET(self):
        if not self._trusted():
            return self._send(403, {"error": "forbidden"})
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            if not SITE.exists():
                return self._send(500, "site/index.html 不存在，先运行 uv run build.py", "text/plain; charset=utf-8")
            return self._send(200, SITE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/ping":
            return self._send(200, versions())
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/run":
            return self._send(404, {"error": "not found"})
        if not self._trusted() or self.headers.get("X-ReviewHelper") != "run":
            return self._send(403, {"error": "forbidden"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            return self._send(413, {"error": "代码太长"})
        try:
            job = json.loads(self.rfile.read(n).decode("utf-8"))
            slug, code = job["slug"], job["code"]
            assert isinstance(slug, str) and isinstance(code, str)
        except Exception:  # noqa: BLE001
            return self._send(400, {"error": "请求格式不对"})
        if find_exercise(slug) is None:
            return self._send(404, {"error": f"找不到题目 {slug}"})
        t0 = time.perf_counter()
        res = run_in_subprocess(slug, code)
        tests = res.get("tests") or []
        npass = sum(t["status"] == "pass" for t in tests)
        state = "超时" if res.get("timeout") else "导入失败" if res.get("import_error") else f"{npass}/{len(tests)} 通过"
        print(f"  运行 {slug:<24} {state:<10} {time.perf_counter() - t0:.1f}s", flush=True)
        return self._send(200, res)

    def log_message(self, fmt, *args):  # 安静一点：只打印运行记录
        pass


def check():
    """CI 自检：参考答案应全部通过，模板应全部是 TODO。"""
    ok = True
    for slug in ("gae", "lru_cache"):
        d = find_exercise(slug)
        for which in ("solution", "template"):
            r = run_in_subprocess(slug, (d / f"{which}.py").read_text(encoding="utf-8"))
            sts = [t["status"] for t in r.get("tests", [])]
            if which == "solution":
                good = bool(sts) and all(s == "pass" for s in sts)
            else:  # 模板：没有失败，且至少一个 TODO（个别用例对空实现也成立）
                good = "todo" in sts and not {"fail", "error"} & set(sts)
            good = good and not r.get("import_error")
            ok &= good
            print(f"{'OK ' if good else 'BAD'} {slug} {which}: {sts or r}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="ReviewHelper 本地练习服务")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    ap.add_argument("--check", action="store_true", help="自检后退出")
    ap.add_argument("--job", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.job:
        return job_main()
    if args.check:
        return check()

    sys.path.insert(0, str(ROOT))
    import build  # 每次启动先重新构建，保证网页和本地题目一致
    build.main()

    try:
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        sys.exit(f"端口 {args.port} 用不了（{e.strerror}）。换一个：uv run serve.py --port 8001")
    url = f"http://localhost:{args.port}/"
    v = versions()
    print(f"\n本地练习服务已启动：{url}")
    print(f"Python {v['python']} · PyTorch {v['torch'] or '未安装（先 uv sync）'} · 按 Ctrl+C 退出\n")
    if not args.no_open:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
