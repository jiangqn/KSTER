import os
import glob
import subprocess
import yaml

subfolder = os.listdir("models")[0]
folder = os.path.join("models", subfolder)
files = glob.glob("%s/[0-9]*.ckpt" % folder)
ids = sorted([int(f[len(folder)+1:-5]) for f in files])

config_path = glob.glob("*.yaml")[0]
config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

keep_last_ckpts = config["combiner_training"]["keep_last_ckpts"]
ids = ids[0:keep_last_ckpts]

inputs_str = " ".join(["%s/%d.ckpt" % (folder, id) for id in ids])
output_str = "%s/averaged.ckpt" % folder

subprocess.call("python3 scripts/average_checkpoints.py --inputs %s --output %s" % (inputs_str, output_str), shell=True)
subprocess.call("cp models/%s/best.ckpt ." % subfolder, shell=True)
subprocess.call("cp models/%s/averaged.ckpt ." % subfolder, shell=True)