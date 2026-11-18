#!/usr/bin/env python3
"""
Phone alerts via ntfy (same setup as the CFB model; can share the same topic
or use a new one). One notification per run, only when something changed:
a new 🔥 A play (spread, total or prop) or an A play that is no longer A.
Topic comes from the NTFY_TOPIC secret. --test sends a test ping.
"""
import base64
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

LOG = "docs/data/play_log.json"


def _dt(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def label(p, h):
    line = h[1]
    if p["market"] == "spread":
        team = p["home"] if p["side"] == "home" else p["away"]
        ln = line if p["side"] == "home" else -line
        return f"{team} {ln:+g}"
    if p["market"] == "total":
        return f"{'U' if p['side'] == 'under' else 'O'}{line:g} {p['away']} at {p['home']}"
    return f"{p['player']} {'O' if p['side'] == 'over' else 'U'}{line:g} {str(p.get('prop_market')).lower()}"


def changes(now):
    if not os.path.exists(LOG):
        return [], []
    log = json.load(open(LOG))
    run = log.get("updated")
    isA = lambda h: h is not None and h[4] == "A" and h[5]
    new, lost = [], []
    for p in log.get("plays", {}).values():
        h = p.get("h") or []
        k = _dt(p.get("kickoff"))
        if not h or (k and k <= now):
            continue
        last = h[-1]
        this = last[0] == run
        prev = h[-2] if this and len(h) > 1 else (None if this else last)
        if this and isA(last) and not isA(prev):
            new.append(f"🔥 {p['market'].upper()}: {label(p, last)}")
        elif isA(prev) and not (this and isA(last)):
            lost.append(f"❌ {label(p, last)}: {'now ' + last[4] if this and last[5] else 'no longer official'}")
    return new, lost


def send(topic, title, body, click):
    req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=body.encode(), method="POST")
    req.add_header("Title", "=?UTF-8?B?" + base64.b64encode(title.encode()).decode() + "?=")
    req.add_header("Tags", "basketball")
    req.add_header("Click", click)
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status


def main():
    test = "--test" in sys.argv
    topic = (os.environ.get("NTFY_TOPIC") or "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "jaredpatchett/CBB-Model").split("/")
    click = f"https://{repo[0].lower()}.github.io/{repo[1]}/dashboard.html"
    if test:
        title, body = "CBB alerts are on ✅", "You'll get a ping when a 🔥 A play joins or drops."
    else:
        new, lost = changes(datetime.now(timezone.utc))
        if not (new or lost):
            print("Alerts: nothing changed")
            return
        title = "CBB: " + ", ".join(x for x in (f"{len(new)} new 🔥 A" if new else "", f"{len(lost)} dropped" if lost else "") if x)
        body = "\n".join((new + lost)[:14])
    print(title, "\n", body)
    if topic and "--dry-run" not in sys.argv:
        print("sent", send(topic, title, body, click))
    elif not topic:
        print("[note] NTFY_TOPIC not set -- not sent")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[note] alerts skipped ({e})")
