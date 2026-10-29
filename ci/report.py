"""Turn a job's result directory into a Markdown summary, and stage the public artifact.

Only result.json, meta.json and log tails go public, after a scrub for anything that looks
like a host name, IP address or home directory.
"""

import json
import pathlib
import re
import shutil
import sys

SCRUB = [
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<ip>"),
    (re.compile(r"/home/[^\s/]+"), "/home/<user>"),
    (re.compile(r"/tmp/tmp\.[A-Za-z0-9]+"), "/tmp/<dir>"),
]


def scrub(text: str) -> str:
    for rx, repl in SCRUB:
        text = rx.sub(repl, text)
    return text


def main(res_dir: str) -> None:
    res = pathlib.Path(res_dir)
    meta = json.loads((res / "meta.json").read_text()) if (res / "meta.json").exists() else {}
    result = json.loads((res / "result.json").read_text()) if (res / "result.json").exists() else {"steps": {}}
    icon = {"pass": "✅", "fail": "❌", "skip": "⏭️", "absent": "—"}
    print(f"### {meta.get('arch', result.get('arch', '?'))} / {meta.get('line', '?')}\n")
    print(f"upstream base `{meta.get('base_sha', '?')}`" + (f", PRs applied: {' '.join(meta['prs'])}" if meta.get("prs") else "") + f", {meta.get('utc', '')}\n")
    print("| step | status | detail |\n|---|---|---|")
    for k, v in result.get("steps", {}).items():
        print(f"| {k} | {icon.get(v['status'], v['status'])} | {scrub(v.get('detail', ''))} |")
    pub = res / "public"
    pub.mkdir(exist_ok=True)
    for name in ("meta.json", "result.json"):
        if (res / name).exists():
            (pub / name).write_text(scrub((res / name).read_text()))
    if (res / "logs").is_dir():
        (pub / "logs").mkdir(exist_ok=True)
        for f in (res / "logs").iterdir():
            (pub / "logs" / f.name).write_text(scrub(f.read_text(errors="replace")))


if __name__ == "__main__":
    main(sys.argv[1])
