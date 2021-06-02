import glob
import pdb
import os
import argparse

parser = argparse.ArgumentParser(description='Correct paths to rgb and depth')
parser.add_argument("--pinhole_path",
                        required=True,
                        help="Path to pinhole folder")
parser.add_argument("--align_mode",
                        required=True,
                        help="Name of semantics directory: PV or labels")

args = parser.parse_args()

# basepath = '/mnt/t4/data/HL2/2021-04-22-184943/pinhole_projection'
basepath = args.pinhole_path
rgbfiles = glob.glob(os.path.join(basepath,args.align_mode,'*'))
depthfiles = glob.glob(os.path.join(basepath,'depth','*'))
rgbfiles = sorted(rgbfiles)
depthfiles = sorted(depthfiles)
with open(os.path.join(basepath,f'{args.align_mode}.txt'), 'w') as filehandle:
    filehandle.writelines("%s\n" % place for place in rgbfiles)
with open(os.path.join(basepath,'depth.txt'), 'w') as filehandle:
    filehandle.writelines("%s\n" % place for place in depthfiles)

