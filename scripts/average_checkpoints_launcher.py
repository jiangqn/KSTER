import os
import glob
import subprocess

subfolder = os.listdir("models")[0]
folder = os.path.join("models", subfolder)
inputs_str = " ".join(glob.glob("%s/[0-9]*.ckpt" % folder))
output_str = "%s/averaged.ckpt" % folder

subprocess.call("python3 scripts/average_checkpoints.py --inputs %s --output %s" % (inputs_str, output_str), shell=True)