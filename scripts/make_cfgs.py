"""Derive Avenue / ShanghaiTech cfgs from the ped2 cfgs.
Usage: python make_cfgs.py <dataset> <src_cfgs_dir> <out_cfgs_dir>
  dataset in {avenue, shanghaitech}
Paper settings (sec 4.2):
  memory modules: ped2=3, avenue=3, shanghaitech=2  -> mem_usage
  w_r,w_p: ped2=(1.0,0.1), avenue=(0.05,1.0), shanghaitech=(0.02,1.0)
  batch/epoch: ped2=(128,80)? avenue=(128,80), shanghaitech=(256,50)
"""
import yaml, os, sys

ds = sys.argv[1]
src = sys.argv[2]
out = sys.argv[3]
os.makedirs(out, exist_ok=True)

WR = {"avenue": 0.05, "shanghaitech": 0.02}[ds]
WP = {"avenue": 1.0, "shanghaitech": 1.0}[ds]
# memory modules: avenue=3 (same as ped2), shanghaitech=2
MEM_USAGE = {"avenue": [False, True, True, True],
             "shanghaitech": [False, False, True, True]}[ds]
SKIP_OPS = {"avenue": ["none", "concat", "concat"],
            "shanghaitech": ["none", "none", "concat"]}[ds]  # official sh cfg
# epochs (keep 80; shanghaitech paper uses 50)
NUM_EPOCHS = {"avenue": 80, "shanghaitech": 50}[ds]


def load(name):
    with open(os.path.join(src, name)) as f:
        return yaml.safe_load(f)


def dump(c, name):
    with open(os.path.join(out, name), "w") as f:
        yaml.safe_dump(c, f, default_flow_style=False, sort_keys=False)
    print("wrote", os.path.join(out, name))


# ---- ml_memAE_sc_cfg.yaml (flow reconstruction stage) ----
c = load("ml_memAE_sc_cfg.yaml")
c["dataset_name"] = ds
c["exp_name"] = "%s_ML_MemAE_SC" % ds
c["model_paras"]["mem_usage"] = MEM_USAGE
c["model_paras"]["skip_ops"] = SKIP_OPS
c["num_epochs"] = NUM_EPOCHS
dump(c, "ml_memAE_sc_cfg.yaml")

# ---- cfg.yaml (CVAE stage) ----
c = load("cfg.yaml")
c["dataset_name"] = ds
c["exp_name"] = "%s_ML_MemAE_SC_CVAE" % ds
c["model_paras"]["mem_usage"] = MEM_USAGE
c["model_paras"]["skip_ops"] = SKIP_OPS
c["ML_MemAE_SC_pretrained"] = "./ckpt/%s_ML_MemAE_SC/best.pth" % ds
c["w_r"] = WR
c["w_p"] = WP
c["num_epochs"] = NUM_EPOCHS
dump(c, "cfg.yaml")

# ---- finetune_cfg.yaml ----
c = load("finetune_cfg.yaml")
c["dataset_name"] = ds
c["exp_name"] = "%s_ML_MemAE_SC_CVAE_finetune" % ds
c["model_paras"]["mem_usage"] = MEM_USAGE
c["model_paras"]["skip_ops"] = SKIP_OPS
c["pretrained"] = "./ckpt/%s_ML_MemAE_SC_CVAE/best.pth" % ds
c["w_r"] = WR
c["w_p"] = WP
c["num_epochs"] = NUM_EPOCHS
dump(c, "finetune_cfg.yaml")
print("DONE for", ds)
