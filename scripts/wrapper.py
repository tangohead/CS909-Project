import subprocess
import os 

args = []
cwd = os.getcwd()
for i in range(2,7):
	#for j in range(1,4):
	args.append(["python","load_data.py", str(i), "1", "0", "0"])

print args
print cwd

for i in args:
	print "*#"*50
	print "\n"
	print "RUNNING ARGS " + str(i)
	print "\n"
	print "*#"*50
	subprocess.check_call(i)