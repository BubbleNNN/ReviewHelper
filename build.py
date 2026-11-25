#!/usr/bin/env python3
"""把 content/ 和 exercises/ 打包成单文件网页 site/index.html。

    uv run build.py          # 或 python3 build.py（只用标准库）

网页模板在 site/src/template.html，数据以 JSON 注入到 <!--RH_DATA--> 处。
"""
import datetime
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
TEMPLATE = ROOT / "site" / "src" / "template.html"
OUT = ROOT / "site" / "index.html"
ARTIFACT = ROOT / "site" / "artifact.html"
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SPLIT_MARK = "<!-- solution -->"


def parse_front_matter(text):
    meta = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            text = text[end + 4:].lstrip("\n")
    if "tags" in meta:
        meta["tags"] = [t.strip() for t in meta["tags"].split(",") if t.strip()]
    if "order" in meta:
        meta["order"] = float(meta["order"])
    return meta, text


def make_id(sec_id, stem):
    i = f"{sec_id}-{stem.replace('_', '-').lower()}"
    if not ID_RE.match(i):
        sys.exit(f"非法 id（只能用小写字母、数字、连字符）: {i}")
    return i


def load_docs(sec):
    items = []
    for p in sorted((ROOT / sec["dir"]).glob("*.md")):
        meta, body = parse_front_matter(p.read_text(encoding="utf-8"))
        items.append({
            "id": make_id(sec["id"], p.stem),
            "title": meta.get("title", p.stem),
            "summary": meta.get("summary", ""),
            "tags": meta.get("tags", []),
            "order": meta.get("order", 999),
            "md": body,
            "path": str(p.relative_to(ROOT)),
        })
    return items


def load_exercises(sec):
    items = []
    base = ROOT / sec["dir"]
    for d in sorted(x for x in base.iterdir() if x.is_dir()) if base.exists() else []:
        prob = d / "problem.md"
        if not prob.exists():
            continue
        meta, body = parse_front_matter(prob.read_text(encoding="utf-8"))
        statement, _, explain = body.partition(SPLIT_MARK)
        tests = sorted(d.glob("test_*.py"))
        items.append({
            "id": make_id(sec["id"], d.name),
            "slug": d.name,
            "title": meta.get("title", d.name),
            "difficulty": meta.get("difficulty", ""),
            "tags": meta.get("tags", []),
            "order": meta.get("order", 999),
            "source": meta.get("source", ""),
            "md": statement.strip(),
            "explain": explain.strip(),
            "template": (d / "template.py").read_text(encoding="utf-8"),
            "solution": (d / "solution.py").read_text(encoding="utf-8"),
            "tests": [{"name": t.name, "code": t.read_text(encoding="utf-8")} for t in tests],
            "path": str(d.relative_to(ROOT)),
        })
    return items


def load_qa(sec):
    items = []
    for p in sorted((ROOT / sec["dir"]).glob("*.md")):
        meta, body = parse_front_matter(p.read_text(encoding="utf-8"))
        cards = []
        for chunk in re.split(r"(?m)^## ", "\n" + body)[1:]:
            q, _, a = chunk.partition("\n")
            cards.append({"q": q.strip(), "a": a.strip()})
        items.append({
            "id": make_id(sec["id"], p.stem),
            "title": meta.get("title", p.stem),
            "summary": meta.get("summary", ""),
            "tags": meta.get("tags", []),
            "order": meta.get("order", 999),
            "cards": cards,
            "path": str(p.relative_to(ROOT)),
        })
    return items


LOADERS = {"doc": load_docs, "exercise": load_exercises, "qa": load_qa}


def main():
    sections = json.loads((ROOT / "content" / "sections.json").read_text(encoding="utf-8"))
    seen = set()
    for sec in sections:
        sec["items"] = sorted(LOADERS[sec["kind"]](sec), key=lambda x: (x["order"], x["title"]))
        for it in sec["items"]:
            if it["id"] in seen:
                sys.exit(f"重复 id: {it['id']}")
            seen.add(it["id"])
    site_cfg_path = ROOT / "content" / "site.json"
    site_cfg = json.loads(site_cfg_path.read_text(encoding="utf-8")) if site_cfg_path.exists() else {}
    repo = (site_cfg.get("repo") or "").rstrip("/")
    data = {
        "generated": datetime.date.today().isoformat(),
        "site": {"repo": repo, "branch": site_cfg.get("branch") or "main"},
        "sections": sections,
    }
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    html = TEMPLATE.read_text(encoding="utf-8")
    if "<!--RH_DATA-->" not in html:
        sys.exit("模板里缺少 <!--RH_DATA--> 占位符")
    html = html.replace("<!--RH_DATA-->", f'<script id="rh-data" type="application/json">{payload}</script>')
    # artifact.html：发布用的页面片段（发布时会自动包上 doctype/head/body）
    ARTIFACT.write_text(html, encoding="utf-8")
    # index.html：本地直接打开用的完整文档
    OUT.write_text(
        '<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "</head>\n<body>\n" + html + "\n</body>\n</html>\n",
        encoding="utf-8",
    )
    counts = ", ".join(f"{s['title']} {len(s['items'])}" for s in sections)
    print(f"已生成 {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.0f} KB)\n{counts}")


if __name__ == "__main__":
    main()
