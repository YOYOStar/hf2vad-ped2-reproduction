#!/bin/bash
source ~/slurm-env.sh 2>/dev/null
echo "waiting for seed jobs (hf2vad-s*) ..."
while squeue -u "$USER" -h -o %j 2>/dev/null | grep -q '^hf2vad-s'; do sleep 30; done
echo "===== ALL SEEDS DONE ====="
python3 - <<'PY'
import glob, re, os, statistics
d=os.path.expanduser("~/scratch/runs/hf2vad")
rows=[]
for s in [1,2,3,4,5]:
    files=sorted(glob.glob(os.path.join(d, "seed_hf2vad-s%d_*.out"%s)), key=os.path.getmtime)
    if not files:
        rows.append([s,None,None,None,"NO_LOG"]); continue
    txt=open(files[-1],errors="ignore").read()
    best=re.findall(r"Best AUC\s+([0-9.]+)", txt)
    st1=float(best[0]) if len(best)>=1 else None
    st2=float(best[1]) if len(best)>=2 else None
    m=re.search(r"SEEDRESULT seed=\d+ finetuned_auc=([0-9.]+)", txt)
    fin=float(m.group(1)) if m else None
    note="ERR" if (fin is None and "Traceback" in txt) else ("RUNNING?" if fin is None else "")
    rows.append([s,st1,st2,fin,note])

print("seed | stage1 ML-MemAE-SC | stage2 CVAE | finetuned | note")
for s,a,b,c,n in rows:
    print("  %s  |  %s  |  %s  |  %s  | %s" % (s,a,b,c,n))

fin=[r[3] for r in rows if r[3] is not None]
cv =[r[2] for r in rows if r[2] is not None]
if fin:
    print("\nFINETUNED AUC: n=%d mean=%.4f std=%.4f min=%.4f max=%.4f"%(
        len(fin), statistics.mean(fin), statistics.pstdev(fin), min(fin), max(fin)))
if cv:
    print("CVAE     AUC: n=%d mean=%.4f std=%.4f"%(len(cv), statistics.mean(cv), statistics.pstdev(cv)))
print("\nfinetuned - CVAE per seed (negative => finetune退化):")
for r in rows:
    if r[2] is not None and r[3] is not None:
        print("  seed %s: %+.4f"%(r[0], r[3]-r[2]))
PY
echo "===== COLLECT_END ====="
