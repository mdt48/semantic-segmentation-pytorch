
import os, glob


file_list = [f for i,f in enumerate(glob.glob("/pless_nfs/home/datasets/Hotels-50K/images/test/unoccluded/0/**/*.jpg", recursive=True)) if "SEG" not in f and i < 100]


for img in file_list[:25]:
    bashCommand = "python -u test.py --imgs {} --gpu 0 --cfg config/hotels_exp15.yaml "
    bashCommand = bashCommand.format(img)
    print(bashCommand)
    import subprocess
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

