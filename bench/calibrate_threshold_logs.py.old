"""Choose RECOGNITION_THRESHOLD from the two real walkthroughs.

  walk1 = a NOT-enrolled person  -> every observation is an impostor
  walk2 = enrolled people        -> the high-scoring observations are genuine

Also simulates the 3-of-5 track vote, which is the real protection: a stranger
must produce three frames above threshold that agree on the SAME identity.
"""
import re, sys
from collections import defaultdict, Counter
import numpy as np

def parse(path):
    out = []
    for line in open(path, errors="ignore"):
        m = re.search(r"^(\S+)\s+t(\d+)\s+(\d+)px sh\s*(\d+) yaw\s*(-?\d+\.\d) pit\s*(-?\d+\.\d) "
                      r"al(\d\.\d+)\s+(\S+) best=\s*(-?\d\.\d+) 2nd=\s*(-?\d\.\d+) (.*)$", line)
        if not m: continue
        out.append(dict(cam=m[1], tid=int(m[2]), px=int(m[3]), sh=int(m[4]),
                        yaw=float(m[5]), pit=float(m[6]), al=float(m[7]),
                        gate=m[8], best=float(m[9]), second=float(m[10]), who=m[11].strip()))
    return out

def gated(r):
    return r["px"] >= 50 and r["sh"] >= 25 and r["al"] >= 0.40 and abs(r["yaw"]) <= 45 and abs(r["pit"]) <= 40

imp = [r for r in parse("/tmp/walk.log")  if gated(r)]
gen = [r for r in parse("/tmp/walk2.log") if gated(r)]
print(f"impostor run (not enrolled): {len(imp)} gate-passing observations")
print(f"genuine  run (enrolled)    : {len(gen)} gate-passing observations\n")

ib = np.array([r["best"] for r in imp]); gb = np.array([r["best"] for r in gen])
print(f"impostor best-score: max={ib.max():.3f}  p99={np.percentile(ib,99):.3f}  median={np.median(ib):.3f}")
print(f"genuine  best-score: max={gb.max():.3f}  p90={np.percentile(gb,90):.3f}  median={np.median(gb):.3f}\n")

def vote_sim(rows, thr, window=5, required=3):
    """Per (cam,track): would 3-of-5 agreement commit, and to whom?"""
    tracks = defaultdict(list)
    for r in rows:
        tracks[(r["cam"], r["tid"])].append(r)
    commits = {}
    for key, obs in tracks.items():
        votes = []
        for r in obs:
            votes.append(r["who"] if r["best"] >= thr else None)
            w = votes[-window:]
            c = Counter(v for v in w if v)
            if c and c.most_common(1)[0][1] >= required:
                commits[key] = c.most_common(1)[0][0]
                break
    return commits, len(tracks)

print(f"{'thr':>6} {'imp obs>=thr':>13} {'gen obs>=thr':>13} | {'FALSE commits':>14} {'TRUE commits':>13}  who")
print("-"*95)
for thr in (0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40, 0.45):
    fc, ntf = vote_sim(imp, thr)
    tc, ntg = vote_sim(gen, thr)
    names = ", ".join(sorted(set(tc.values())))[:38]
    print(f"{thr:6.2f} {int((ib>=thr).sum()):6d}/{len(ib):<6} {int((gb>=thr).sum()):6d}/{len(gb):<6} | "
          f"{len(fc):6d}/{ntf:<7} {len(tc):6d}/{ntg:<6}  {names}")

print("\nrecommendation: highest threshold that still commits the genuine tracks,")
print("with zero false commits on the impostor run.")
