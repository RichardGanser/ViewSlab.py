import os
from typing import TextIO
from pathlib import Path
paths=[]
slabs_dir= ["s29_folder", "s137_folder"]
slab_path= r"interphases/Minislabs/aims/"
print(os.listdir(slab_path))
with open("Paths_to_Minislabs.txt", "w") as file, open("Paths_to_Minislabs_Runfiles.txt", "w") as runfile:
	for path in os.listdir((slab_path)):
		for dir in slabs_dir:
			print(path)
			file.write(str(Path(str(slab_path) + str(path) + "/"+str(dir) + "\n")))
			runfile.write(str(slab_path) + str(path) + "/"+str(dir) +"/run_aims_cm2_tiny_1nodes\n")



"""with open("Paths_to_Minislabs.txt", "a") as file, open("Paths_to_Minislabs_Runfiles.txt", "a") as runfile:
	for dirname in os.listdir(slab_path):
		if dirname[-1] == str(3):
			file.write(slab_path+dirname+"\n")
			runfile.write(slab_path+dirname+"/run_aims_cm2_tiny_4nodes\n")"""