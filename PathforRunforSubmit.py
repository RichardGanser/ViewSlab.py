import os

paths=[]
f = open("Paths_to_Minislabs_Runfiles.txt","a")

path = r"interphases/Minislabs/"
for filename in os.listdir(path):
	print(filename[-1])
	if filename[-1]==str(3):
	#if '[ 1.  3. -2.]' in filename:
		print(filename)
		Path2 = os.path.join(path,filename)
		f.write(str(os.path.join(Path2,"run_cm2__std_8nodes \n"))) 
#print(os.path.join(Path,i))
#f.write(str(os.path.join(Path,i))) 
f.close()