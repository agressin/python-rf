#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: agressin
"""

import paramiko
import os
import sys, getopt
import argparse



########################################################################
# ls
########################################################################
def ssh_ls(ssh,filename,show=True):
		stdin, stdout, stderr = ssh.exec_command('ls -l ' + filename)
		exist = False
		for line in stdout:
			if(show):
				print(line)
			exist = True
		
		return exist

########################################################################
# tail
########################################################################
def ssh_tail(ssh,filename, tail = 0, grep = ""):
	
	if grep:
		cmd="cat "
	else:
		cmd="tail "
		if tail:
			cmd += "-n "+ str(tail)
	
	cmd += " "+ filename +" "
	if grep:
		cmd += "| grep " + grep
		if tail:
			cmd += " | tail -n "+ str(tail)
	print(cmd)
	stdin, stdout, stderr = ssh.exec_command(cmd)
	for line in stdout:
			print(line)

########################################################################
# progress bar
########################################################################
def progress(val, valmax):
	# format
	if val > valmax: val = valmax

	maxbar=100    
	# process
	perc  = round((float(val) / float(valmax)) * 100)
	scale = 100.0 / float(maxbar)
	bar   = int(perc / scale)

	# render 
	out = '\r Progress [%s%s] %3d %%' % ('=' * bar, ' ' * (maxbar - bar), perc)
	sys.stdout.write(out)

########################################################################
# get
########################################################################
def ssh_get(ssh,filename_dist,filename_local):
		#On test si le fichier existe sur le dist
		dist = ssh_ls(ssh,filename_dist,False)

		#Et sur le local
		local = os.path.isfile(filename_local)
		if dist and (not local) :
			sftp = ssh.open_sftp()
			print(filename_dist)
			sftp.get(filename_dist,filename_local,progress)
			sftp.close()
		if local:
			print("Already have : ",filename_local)
			
		if(not dist and not local):
			print("Still waiting for : ",filename_dist)

########################################################################
# is_busy
########################################################################
def is_busy(ssh,host,username="prof",show = True):
	ssh.connect(host, username=username)
	busy = False
	#On test si le fichier existe sur le dist
	pid="/home/prof/iPython/tmp.pid"
	if ssh_ls(ssh,pid,False):
		stdin, stdout, stderr = ssh.exec_command("ps up `cat "+pid+" `  && echo 'Run' || echo 'Stop' ")
		for line in stdout:
			if show:
				if(line.startswith("prof")):
					tt = line.split()
					print(host,"cpu", tt[2], "mem", tt[3],"start", tt[8],"time", tt[9] )
			busy = ("Run" in line)
		if not busy : #tmp.pif exist but process don't exist anymore
			print(host, "rm "+pid)
			ssh.exec_command("rm "+pid)
			
	return busy

########################################################################
# get_hosts
########################################################################
def get_hosts(filename_hosts):
	if(os.path.isfile(filename_hosts)):
		with open(filename_hosts) as f:
			lines = f.read().splitlines()
			
	lines = list(filter(None, lines))
	return lines

########################################################################
# Parameters
########################################################################

def main(argv):

	parser = argparse.ArgumentParser(description="Run all over ssh.")

	parser.add_argument('-r', '--run', help='Run', action='store_true', default=False)
	parser.add_argument('-g', '--get', help='Get', action='store_true', default=False)
	parser.add_argument('-l', '--ls', help='ls', action='store_true', default=False)
	parser.add_argument('-t', '--tail', help='Tail', type=int, default=0)
	parser.add_argument('-tg', '--tail_grep', help='tail_grep', type=str, default="")
	parser.add_argument('-f', '--filename_hosts', help='Hosts', type=str, default="")
	
	args = parser.parse_args()

	run		= args.run
	get		= args.get
	ls		= args.ls
	tail	= args.tail
	tail_grep	= args.tail_grep
	filename_hosts = args.filename_hosts


	#server	n_tree	max_feature	max_depth	n_forest	Fusion
	tmp_1=[
		["rks1012w035.ensg.eu" , "5" , "20" , "5" , "5" , "Mean"],
		["rks1012w036.ensg.eu" , "5" , "20" , "5" , "10" , "Mean"],
		["rks1012w037.ensg.eu" , "5" , "20" , "5" , "20" , "Mean"],
		["rks1012w038.ensg.eu" , "10" , "20" , "5" , "5" , "Mean"],
		["rks1012w039.ensg.eu" , "10" , "20" , "5" , "10" , "Mean"],
		["rks1012w040.ensg.eu" , "10" , "20" , "5" , "20" , "Mean"],
		["rks1012w041.ensg.eu" , "20" , "20" , "5" , "5" , "Mean"],
		["rks1012w042.ensg.eu" , "20" , "20" , "5" , "10" , "Mean"],
		["rks1012w043.ensg.eu" , "20" , "20" , "5" , "20" , "Mean"],
		["rks1012w044.ensg.eu" , "5" , "200" , "5" , "5" , "Mean"],
		["rks1012w067.ensg.eu" , "5" , "200" , "5" , "10" , "Mean"],
		["rks1012w069.ensg.eu" , "5" , "200" , "5" , "20" , "Mean"],
		["rks1012w070.ensg.eu" , "10" , "200" , "5" , "5" , "Mean"],
		["rks1012w071.ensg.eu" , "10" , "200" , "5" , "10" , "Mean"],
		["rks1012w072.ensg.eu" , "10" , "200" , "5" , "20" , "Mean"],
		["rks1012w073.ensg.eu" , "20" , "200" , "5" , "5" , "Mean"],
		["rks1012w075.ensg.eu" , "20" , "200" , "5" , "10" , "Mean"],
		["rks1012w076.ensg.eu" , "20" , "200" , "5" , "20" , "Mean"]
	]
	tmp_2=[
		["rks1012w035.ensg.eu" , "50"  , "5" , "5"  , "2"  , "Mean"],
		["rks1012w036.ensg.eu" , "50"  , "5" , "10" , "2" , "Mean"],
		["rks1012w037.ensg.eu" , "50"  , "5" , "5"  , "4" , "Mean"],
		["rks1012w038.ensg.eu" , "50"  , "5" , "10" , "4"  , "Mean"],
		["rks1012w039.ensg.eu" , "100" , "5" , "5"  , "2" , "Mean"],
		["rks1012w040.ensg.eu" , "100" , "5" , "10" , "2" , "Mean"],
		["rks1012w041.ensg.eu" , "100" , "5" , "5"  , "4"  , "Mean"],
		["rks1012w042.ensg.eu" , "100" , "5" , "10" , "4" , "Mean"],
		["rks1012w043.ensg.eu" , "50"  , "10", "5"  , "2" , "Mean"],
		["rks1012w044.ensg.eu" , "50"  , "10", "10" , "2"  , "Mean"],
		["rks1012w045.ensg.eu" , "50"  , "5" , "5"  , "1" , "Mean"],################
		["rks1012w067.ensg.eu" , "50"  , "10", "5"  , "4" , "Mean"],
		["rks1012w069.ensg.eu" , "50"  , "10", "10" , "4" , "Mean"],
		["rks1012w070.ensg.eu" , "100" , "10", "5"  , "2"  , "Mean"],
		["rks1012w071.ensg.eu" , "100" , "10", "10" , "2" , "Mean"],
		["rks1012w072.ensg.eu" , "100" , "10", "5"  , "4" , "Mean"],
		["rks1012w073.ensg.eu" , "100" , "10", "10" , "4"  , "Mean"]
	]

	all_params = tmp_2
	ssh = paramiko.SSHClient()
	ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

	if args.filename_hosts:
		hosts = get_hosts(filename_hosts)
		hosts_free = [h for h in hosts if not is_busy(ssh,h)]
		print(hosts_free)
		exit()

	for param in all_params:
		print(param[0])
	
		ssh.connect(param[0], username="prof")
		sftp = ssh.open_sftp()
		
		filename = "test_jungle-500-"+ param[1] +"-"+ param[2] +"-"+ param[3] +"-50-"+ param[4] +".j"
		
		if run :

			sftp.put("run_jungle.py", "/home/prof/iPython/run_jungle.py")
			cmd = "cd /home/prof/iPython;"
			cmd += "echo '"
			cmd += "python3 -u run_jungle.py"
			cmd += " -ne " + param[1]
			cmd += " -mf " + param[2]
			cmd += " -md " + param[3]
			cmd += " -nf " + param[4]
			cmd += " -fu " + param[5]
			cmd += " --train --pid"
			cmd += " | tee > log_jungle' > tmp.bash ;"
			cmd += "screen -dmS compute bash tmp.bash"
			print(cmd)
			ssh.exec_command(cmd)
	
		if ls :
			ssh_ls(ssh,'/home/prof/iPython/out/test_jungle-500*.j')

		if tail or tail_grep :		
			#On test si le fichier existe sur le dist
			if( not ssh_ls(ssh,'/home/prof/iPython/out/'+filename,False) ):
				print("Still waiting for :" ,filename)
				ssh_tail(ssh,"/home/prof/iPython/log_jungle",tail,tail_grep)
			
		if get :
			ssh_get(ssh,'/home/prof/iPython/out/'+filename,'/home/agressin/tt/'+filename)

		sftp.close()
		ssh.close()

########################################################################
# Main
########################################################################
if __name__ == "__main__" :
   main(sys.argv)

