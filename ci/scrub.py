"""Filter for job output: drop host-specific details before they reach the public log."""
import re
import sys

RULES = [
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<ip>"),
    (re.compile(r"/home/[^\s/'\"]+(/[^\s'\"]*)?"), "<path>"),
]
for line in sys.stdin:
    for rx, repl in RULES:
        line = rx.sub(repl, line)
    sys.stdout.write(line)
    sys.stdout.flush()
