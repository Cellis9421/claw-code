#!/usr/bin/env python3
"""Check that no job on the self-hosted keck pool can start for a fork pull request.

Usage:
    python .github/scripts/check_fork_pr_guard.py [WORKFLOW ...]   # default: .github/workflows/*.yml

Why (KK-META#3011). This repo is public. A `pull_request` run from a fork executes the fork's
code, and a job on `[self-hosted, ...]` would execute it on our runner pool, which also runs our
private repos' jobs. So every self-hosted job in a workflow a pull request can trigger carries a
job-level guard that is true for every trusted event and false for a fork's pull request:

    if: github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository

This script parses each workflow (stdlib only, line-based, no PyYAML), finds each self-hosted job,
and EVALUATES its job-level `if:` against four event payloads: a push, a workflow_dispatch, a
same-repo pull request and a fork pull request. It prints the result per job and exits 1 when a
self-hosted job would start for the fork, or would NOT start for a trusted event (the guard must
not change what gates our own pull requests).

Triggers it refuses outright on a self-hosted workflow, because a guard on `head.repo` does not
describe them: `pull_request_target`, `pull_request_review`, `pull_request_review_comment`,
`issue_comment`, `workflow_run`. Add one only with its own guard and a case here.

Limit, stated because it matters: GitHub runs a fork's `pull_request` with the workflow file from
the pull request's merge commit, so a fork can edit this guard away. The guard stops a fork PR that
leaves the workflows alone (including one a maintainer approves); the repository's fork-PR approval
setting is what stops one that edits them.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = "Cellis9421/claw-code"
UNSUPPORTED = {"pull_request_target", "pull_request_review", "pull_request_review_comment",
               "issue_comment", "workflow_run"}

# name -> (event_name, event payload); github.repository is REPO for all of them.
CASES = {
    "push": ("push", {"ref": "refs/heads/main"}),
    "workflow_dispatch": ("workflow_dispatch", {"inputs": {}}),
    "same-repo PR": ("pull_request", {"pull_request": {"head": {"repo": {"full_name": REPO}}}}),
    "fork PR": ("pull_request", {"pull_request": {"head": {"repo": {"full_name": "stranger/claw-code"}}}}),
}
TRUSTED = {"push", "workflow_dispatch", "same-repo PR"}


# --- a tiny GitHub Actions expression evaluator --------------------------------------------------
# Enough of https://docs.github.com/actions/learn-github-actions/expressions for a guard: string
# and boolean literals, `null`, dotted context paths, `!`, `==`, `!=`, `&&`, `||` and parentheses.
# `==` on strings is case-insensitive and a missing property is null, as in Actions. Anything else
# raises, so an unreadable guard fails the check rather than passing it.

TOKEN = re.compile(r"\s*(?:(?P<str>'(?:[^']|'')*')|(?P<op>==|!=|&&|\|\||!|\(|\))|(?P<word>[A-Za-z_][\w.\-]*))")


def tokenize(expr: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    expr = expr.strip()
    while pos < len(expr):
        m = TOKEN.match(expr, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot read expression at {expr[pos:]!r}")
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
        pos = m.end()
        while pos < len(expr) and expr[pos].isspace():
            pos += 1
    return out


def truthy(v) -> bool:
    return v not in (None, False, "", 0)


def lookup(path: str, ctx: dict):
    if path in ("true", "false"):
        return path == "true"
    if path == "null":
        return None
    cur = ctx
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def equal(a, b) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    return a == b


def evaluate(expr: str, ctx: dict):
    toks = tokenize(expr)
    i = 0

    def peek():
        return toks[i] if i < len(toks) else (None, None)

    def take():
        nonlocal i
        i += 1
        return toks[i - 1]

    def primary():
        kind, val = take()
        if kind == "str":
            return val[1:-1].replace("''", "'")
        if kind == "word":
            return lookup(val, ctx)
        if val == "!":
            return not truthy(primary())
        if val == "(":
            v = or_()
            if take()[1] != ")":
                raise ValueError("unbalanced parentheses")
            return v
        raise ValueError(f"unexpected token {val!r}")

    def cmp_():
        left = primary()
        while peek()[1] in ("==", "!="):
            op = take()[1]
            right = primary()
            left = equal(left, right) if op == "==" else not equal(left, right)
        return left

    def and_():
        left = cmp_()
        while peek()[1] == "&&":
            take()
            right = cmp_()
            left = right if truthy(left) else left
        return left

    def or_():
        left = and_()
        while peek()[1] == "||":
            take()
            right = and_()
            left = left if truthy(left) else right
        return left

    v = or_()
    if i != len(toks):
        raise ValueError(f"trailing tokens in {expr!r}")
    return v


# --- line-based workflow reading (the shape `workflow-lint.py` at the workspace root reads) ------

def strip_comment(line: str) -> str:
    return "" if line.lstrip().startswith("#") else re.sub(r"\s#.*$", "", line)


def triggers(lines: list[str]) -> set[str]:
    """Event names under the top-level `on:` (block map, flow list or a single word)."""
    out: set[str] = set()
    for i, l in enumerate(lines):
        m = re.match(r"^(?:on|'on'|\"on\"|true):\s*(.*)$", strip_comment(l))
        if not m:
            continue
        inline = m.group(1).strip()
        if inline:
            out.update(w for w in re.findall(r"[a-z_]+", inline))
            return out
        for nxt in lines[i + 1:]:
            if nxt.strip() and not nxt.startswith(" ") and not nxt.startswith("#"):
                break
            k = re.match(r"^  ([a-z_]+):", nxt) or re.match(r"^  - ([a-z_]+)\s*$", nxt)
            if k:
                out.add(k.group(1))
        return out
    return out


def jobs(lines: list[str]):
    """(job id, body lines) for each job under the top-level `jobs:`."""
    start = next((i for i, l in enumerate(lines) if re.match(r"^jobs:\s*(#.*)?$", l)), None)
    if start is None:
        return []
    out, cur = [], None
    for i in range(start + 1, len(lines)):
        l = lines[i]
        if l.strip() and not l.startswith(" ") and not l.startswith("#"):
            break
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*(#.*)?$", l)
        if m:
            if cur:
                out.append(cur)
            cur = (m.group(1), [])
        elif cur:
            cur[1].append(l)
    if cur:
        out.append(cur)
    return out


def job_key(body: list[str], key: str) -> str | None:
    """A job-level key's value, with a block list or folded scalar below it joined on one line."""
    for k, l in enumerate(body):
        m = re.match(rf"^    {key}:\s*(.*)$", l)
        if not m:
            continue
        value = [strip_comment(m.group(1)) if not m.group(1).startswith(("'", '"')) else m.group(1)]
        for nxt in body[k + 1:]:
            if re.match(r"^      ", nxt) or not nxt.strip():
                value.append(strip_comment(nxt))
            else:
                break
        return " ".join(v.strip() for v in value if v.strip())
    return None


def guard_expr(raw: str | None) -> str:
    if raw is None:
        return "true"  # no `if:` means the job always starts
    raw = raw.strip()
    if raw[:1] in ("'", '"') and raw[-1:] == raw[:1]:
        raw = raw[1:-1]
    raw = re.sub(r"^[>|][-+]?\s*", "", raw)
    m = re.fullmatch(r"\$\{\{(.*)\}\}", raw.strip(), re.S)
    return m.group(1).strip() if m else raw


def check(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    events = triggers(lines)
    errors: list[str] = []
    print(f"{path.name}: triggers {sorted(events)}")
    for job, body in jobs(lines):
        runs_on = job_key(body, "runs-on") or ""
        if "self-hosted" not in runs_on and "keck" not in runs_on:
            print(f"  {job}: runs-on {runs_on or '(none)'}; not the self-hosted pool, not judged")
            continue
        bad = sorted(events & UNSUPPORTED)
        if bad:
            errors.append(f"{path.name}: job `{job}` is on the self-hosted pool in a workflow triggered by "
                          f"{bad}; this check has no safe guard for those triggers")
        expr = guard_expr(job_key(body, "if"))
        print(f"  {job}: runs-on {runs_on}; if: {expr}")
        for name, (event_name, payload) in CASES.items():
            if event_name not in events:
                continue
            ctx = {"github": {"event_name": event_name, "event": payload, "repository": REPO}}
            try:
                starts = truthy(evaluate(expr, ctx))
            except ValueError as e:
                errors.append(f"{path.name}: job `{job}` guard cannot be evaluated: {e}")
                break
            print(f"    {name:18} -> {'starts' if starts else 'skipped'}")
            if name in TRUSTED and not starts:
                errors.append(f"{path.name}: job `{job}` would be skipped for a {name}; the guard must not "
                              f"change what runs for our own events")
            if name not in TRUSTED and starts:
                errors.append(f"{path.name}: job `{job}` would START on the self-hosted pool for a {name}")
    return errors


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[2]
    files = [Path(a) for a in argv] or sorted((root / ".github" / "workflows").glob("*.y*ml"))
    errors = [e for f in files for e in check(f)]
    for e in errors:
        print(f"fork-pr-guard: {e}", file=sys.stderr)
    print(f"fork-pr-guard: {len(files)} workflow(s), {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
