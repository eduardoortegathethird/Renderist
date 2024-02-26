import tensorflow as tf
import numpy as np
import pandas as pd
import subprocess
import PIL.Image


def convert_parametric(x, y, z):
    sw_l = [1**s for s in x]
    cw_l = [1**s for s in y]
    tvw_l = [abs(10*s) for s in z]
    return sw_l, cw_l, tvw_l

def sim(content_path, gif_weights=True):
    df = pd.read_csv('parametric_xyz.csv')
    sw_l, cw_l, tvw_l = convert_parametric(df['x'], df['y'], df['z'])
    if gif_weights:
       print("MAKING GIF OF THE ACTUAL INPUTS") 
    for style in ["art.jpg", "scream.jpg"]:
        cmd = f"python vgg19_style_transfer.py --content {content_path} --style images/style/{style} --cw {cw_l[0]} --sw {sw_l[0]} --tvw {tvw_l[0]} &"
        print("LAUNCHING THE FOLLOWING")
        print(cmd)
        print("============================")
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

def main():
    sim("images/content/city.jpg")

if __name__=="__main__":
    main()
