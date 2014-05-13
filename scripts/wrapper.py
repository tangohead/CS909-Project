import subprocess
import os 

args = []
cwd = os.getcwd()
for i in range(1,7):
	for j in range(1,4):
		args.append(["python","load_data.py", str(i), str(j), "0", "0"])

print args
print cwd

for i in args[5:]:
	print "*#"*50
	print "\n"
	print "RUNNING ARGS " + str(i)
	print "\n"
	print "*#"*50
	subprocess.check_call(i)