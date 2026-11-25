#!/usr/bin/env python3
"""ReviewHelper 题目运行器（不依赖 pytest，只需要 numpy）。

用法
    uv run run.py                 # 跑全部题目，测试你写的 template.py
    uv run run.py gae ppo         # 只跑名字里含 gae / ppo 的题
    uv run run.py gae -v          # 打印每个用例和报错堆栈
    uv run run.py --solution      # 用参考答案跑（检查测试本身）
    uv run run.py --list          # 列出所有题目

没装 uv 的话，把 `uv run` 换成 `python3`，并先 `pip install numpy`。
"""
import argparse
import importlib.util
import inspect
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parent
EX_DIR = ROOT / "exercises"

USE_COLOR = sys.stdout.isatty()


def c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def find_exercises():
    out = []
    for test_file in sorted(EX_DIR.glob("*/*/test_*.py")):
        d = test_file.parent
        out.append({"category": d.parent.name, "slug": d.name, "dir": d, "test": test_file})
    return out


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_exercise(ex, which, verbose):
    """返回 (status, passed, total, messages)。status ∈ PASS / FAIL / TODO / ERROR"""
    msgs = []
    try:
        impl = load_module(ex["dir"] / f"{which}.py", f"rh_{ex['slug']}_{which}")
    except Exception:
        return "ERROR", 0, 0, [f"导入 {which}.py 失败:\n" + traceback.format_exc()]
    tests_mod = load_module(ex["test"], f"rh_test_{ex['slug']}")
    tests = [(n, f) for n, f in vars(tests_mod).items() if n.startswith("test_") and callable(f)]
    passed, todo = 0, 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            if len(inspect.signature(fn).parameters) >= 1:
                fn(impl)
            else:
                fn()
            passed += 1
            if verbose:
                msgs.append(f"  {c('✓', '32')} {name}  {c(f'{(time.perf_counter()-t0)*1e3:.0f}ms', '2')}")
        except NotImplementedError:
            todo += 1
            if verbose:
                msgs.append(f"  {c('·', '33')} {name}  (TODO: 还没实现)")
        except Exception as e:  # noqa: BLE001
            if verbose:
                tb = traceback.format_exc().rstrip().splitlines()
                msgs.append(f"  {c('✗', '31')} {name}\n      " + "\n      ".join(tb[-12:]))
            else:
                msgs.append(f"  {c('✗', '31')} {name}: {type(e).__name__}: {str(e)[:160]}")
    total = len(tests)
    failed = total - passed - todo
    if passed == total:
        status = "PASS"
    elif failed == 0:
        status = "TODO"  # 还有函数没实现，其余没有报错
    else:
        status = "FAIL"
    return status, passed, total, msgs


def main():
    ap = argparse.ArgumentParser(description="ReviewHelper 题目运行器")
    ap.add_argument("filters", nargs="*", help="题目名关键字（子串匹配）")
    ap.add_argument("-s", "--solution", action="store_true", help="测试参考答案 solution.py")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印每个用例")
    ap.add_argument("--list", action="store_true", help="列出题目")
    args = ap.parse_args()

    exs = find_exercises()
    if args.filters:
        exs = [e for e in exs if any(f.lower() in e["slug"].lower() for f in args.filters)]
    if not exs:
        print("没有匹配的题目。用 --list 看看有哪些。")
        return 2
    if args.list:
        for e in exs:
            print(f"{e['category']:<8} {e['slug']}")
        return 0

    which = "solution" if args.solution else "template"
    sys.path.insert(0, str(ROOT))
    print(c(f"\n运行 {len(exs)} 道题 · 被测文件: {which}.py\n", "1"))
    badge = {"PASS": c(" PASS ", "30;42"), "FAIL": c(" FAIL ", "97;41"),
             "TODO": c(" TODO ", "30;43"), "ERROR": c(" ERR  ", "97;45")}
    summary = {"PASS": 0, "FAIL": 0, "TODO": 0, "ERROR": 0}
    for e in exs:
        status, p, t, msgs = run_exercise(e, which, args.verbose)
        summary[status] += 1
        print(f"{badge[status]} {e['category']}/{e['slug']:<28} {p}/{t}")
        show = msgs if (args.verbose or status in ("FAIL", "ERROR")) else []
        for line in show:
            print(line)
    print()
    print("  ".join(f"{k}: {v}" for k, v in summary.items()))
    if summary["FAIL"] or summary["ERROR"]:
        return 1
    # 参考答案不允许有没实现的函数；练习模式下全量跑时 TODO 不算失败
    if summary["TODO"] and (args.solution or args.filters):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
