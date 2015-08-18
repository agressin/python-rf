#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: agressin
"""

import paramiko, os, sys, getopt, argparse, csv,time, subprocess

global jobs_list
jobs_list = []

########################################################################
# ls
########################################################################
def ssh_ls(ssh,filename,show=True):
		stdin, stdout, stderr = ssh.exec_command('ls ' + filename)
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
# get
########################################################################
def ssh_get_all(ssh,filename_dist,filename_local):

		stdin, stdout, stderr = ssh.exec_command('ls ' + filename_dist)
		exist = False
		for line in stdout:
			dist = line[:-1]

			local = filename_local + dist[23:]
			if( not os.path.isfile(local) ):
				sftp = ssh.open_sftp()
				print(dist)
				print(local)
				sftp.get(dist,local,progress)
				sftp.close()
			else :
				print("Already have : ", local)

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
	lines = ""
	if(os.path.isfile(filename_hosts)):
		with open(filename_hosts) as f:
			lines = f.read().splitlines()
			lines = list(filter(None, lines))
			lines = filter(lambda row: row[0]!='#', lines)

	return lines

########################################################################
# get_jobs
########################################################################
def get_jobs(filename_jobs,local = False):
	with open(filename_jobs) as fin:
		reader=csv.reader(filter(lambda row: row[0]!='#', fin), skipinitialspace=True, quotechar='"',delimiter=' ')
		keys=next(reader)
		print(keys)
		for row in reader:
			job={}
			if len(row) == len(keys):
				for idx, val in enumerate(row):
					job[keys[idx]] = val
				cmd,filename = get_info_from(job,local)
				job['cmd'] = cmd
				job['run'] = False
				job['finish'] = False
				job['host'] = None
				job['filename'] = filename
				jobs_list.append(job)

########################################################################
# get_cmd_from
########################################################################
def get_info_from(job,local):

	if not local:
		cmd = "cd /home/prof/iPython;"
		cmd += "echo '"
		#cmd += "python3 -u run_jungle.py"
		cmd += "run_jungle.py"
	else :
		cmd = "run_jungle.py"


	filename = "jungle"

	if 'image' in job:
		cmd += " -i " + job['image']
	if 'label' in job:
		cmd += " -l " + job['label']
	if 'output' in job:
		cmd += " -o " + job['output']
	if 'nb_samples' in job:
		cmd += " -ns " + job['nb_samples']
		filename += "-ns-" + job['nb_samples']
	else:
		filename += "-ns-500"
	if 'windows_size' in job:
		cmd += " -ws " + job['windows_size']
		filename += "-ws-" + job['windows_size']
	else:
		filename += "-ws-50"
	if 'nb_estimators' in job:
		cmd += " -ne " + job['nb_estimators']
		filename += "-ne-" + job['nb_estimators']
	else:
		filename += "-ne-50"
	if 'max_features' in job:
		cmd += " -mf " + job['max_features']
		filename += "-mf-" + job['max_features']
	else:
		filename += "-mf-20"
	if 'max_depth' in job:
		cmd += " -md " + job['max_depth']
		filename += "-md-" + job['max_depth']
	else:
		filename += "-md-5"

	if 'nb_forests' in job:
		cmd += " -nf " + job['nb_forests']
		filename += "-nf-" + job['nb_forests']
	elif 'nb_steps_simple' in job and 'nb_steps_proba' in job:
		cmd += " -nss " + job['nb_steps_simple']
		cmd += " -nsp " + job['nb_steps_proba']
		filename += "-nss-" + job['nb_steps_simple']
		filename += "-nsp-" + job['nb_steps_proba']
	else:
		filename += "-nf-2"

	if 'specialisation' in job:
		cmd += " -sp " + job['specialisation']
		filename += "-sp-" + job['specialisation']
	else:
		filename += "-sp-none"
	if 'add_previous_prob' in job:
		if job['add_previous_prob'] == "1" or job['add_previous_prob'] == "True":
			cmd += " -app "
			filename += "-app"
	if 'use_geodesic' in job:
		if job['use_geodesic'] == "1" or job['use_geodesic'] == "True":
			cmd += " -ug "
			filename += "-ug"
	if 'fusion' in job:
		cmd += " -fu " + job['fusion']
		filename += "-fu-" + job['fusion']
	else:
		filename += "-fu-last"

	filename += ".j"
	if local :
		cmd += " -oj out/" +filename
	else :
		cmd += " -oj /home/prof/iPython/out/" +filename

	cmd += " --train "

	if not local:
		cmd += " --pid "
		cmd += " | tee > log_jungle' > tmp.bash ;"
		cmd += "screen -dmS compute bash tmp.bash"

	return cmd,filename

########################################################################
# filter_dict_list
########################################################################
def filter_dict_list(dict_list,key,value):
	return [j for j in dict_list if j[key] == value]

########################################################################
# Parameters
########################################################################

def main(argv):

	parser = argparse.ArgumentParser(description="Run all over ssh.")

	parser.add_argument('-r', '--run', help='Run', action='store_true', default=False)
	parser.add_argument('-c', '--clean', help='Clean', action='store_true', default=False)
	parser.add_argument('-g', '--get', help='Get', action='store_true', default=False)
	parser.add_argument('-l', '--ls', help='ls', action='store_true', default=False)
	parser.add_argument('-t', '--tail', help='Tail', type=int, default=0)
	parser.add_argument('-tg', '--tail_grep', help='tail_grep', type=str, default="")
	parser.add_argument('-f', '--filename_hosts', help='Hosts', type=str, default="")
	parser.add_argument('-j', '--filename_jobs', help='Hosts', type=str, default="")

	args = parser.parse_args()

	if len(sys.argv) <= 1 :
		parser.print_help()
		exit()

	run		= args.run
	get		= args.get
	ls		= args.ls
	clean	= args.clean
	tail	= args.tail
	tail_grep	= args.tail_grep
	filename_hosts = args.filename_hosts
	filename_jobs = args.filename_jobs

	#all_params = tmp_1
	ssh = paramiko.SSHClient()
	ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

	hosts_list = get_hosts(filename_hosts)
	#hosts_free = [h for h in hosts if not is_busy(ssh,h)]
	#print(hosts_free)

	if args.filename_hosts:
		print("ssh")
		if ls :
			for h in hosts_list:
				print(h)
				ssh.connect(h, username="prof")
				ssh_ls(ssh,'/home/prof/iPython/out/jungle*.j')
				ssh.close()


		if tail or tail_grep :
			for h in hosts_list:
				if is_busy(ssh,h):
					ssh.connect(h, username="prof")
					ssh_tail(ssh,"/home/prof/iPython/log_jungle",tail,tail_grep)
					ssh.close()

		if get :
			for h in hosts_list:
				ssh.connect(h, username="prof")
				ssh_get_all(ssh,'/home/prof/iPython/out/jungle*.j','/home/agressin/tt/')
				ssh.close()

		if clean :
			print("Are you sure to want remove all .j  in /home/prof/iPython/out ? (y/N)")
			z = input ()
			if z == "y" or z == "Y" or z == "yes" or z == "YES" :
				for h in hosts_list:
					ssh.connect(h, username="prof")
					cmd = 'rm /home/prof/iPython/out/jungle*.j'
					print(cmd)
					ssh.exec_command(cmd)
					ssh.close()

		if run :

			if not args.filename_jobs :
				print("No file jobs")
				exit()

			get_jobs(filename_jobs,local=False)

			is_finish = False
			while not is_finish:
				#1/ lancer jobs sur host free
				print("1/ lancer jobs sur host free")
				hosts_free = [h for h in hosts_list if not is_busy(ssh,h)]
				print("Get ",len(hosts_free),"free hosts")
				for h in hosts_free:
					print(h)
					for j in jobs_list:
						if j['host'] is None:
							j['host'] = h
							j['run'] = True
							ssh.connect(h, username="prof")
							#sftp = ssh.open_sftp()
							#sftp.put("run_jungle.py", "/home/prof/iPython/run_jungle.py")
							#sftp.close()
							cmd = j['cmd']
							print(h,cmd)
							ssh.exec_command(cmd)
							ssh.close()
							break

				#2/ recuperer les fichiers finis
				print("2/ recuperer les fichiers finis")
				still_some_job = False
				for j in jobs_list:
					if j['run'] and j['host'] is not None:
						filename = j['filename']
						ssh.connect(j['host'], username="prof")
						if ssh_ls(ssh,filename,False):
							j['run'] = False
							sftp = ssh.open_sftp()
							ssh_get(ssh,'/home/prof/iPython/out/'+filename,'/home/agressin/tt/'+filename)
							sftp.close()
							j['finish'] = True
						else:
							still_some_job = True
							print(filename,"still running on ",j['host'])
						ssh.close()
				is_finish = not still_some_job
				#3/ compute classif + rapport en local
				#TODO

				#On attend 10 min avant l'étape suivante
				time.sleep(60*10)
			#while
		#run
	else :
		print("local")
		#run locally
		if run :
			get_jobs(filename_jobs,local=True)
			for j in jobs_list:
				cmd = j['cmd']
				print(cmd)
				subprocess.call(cmd, shell=True)





########################################################################
# Main
########################################################################
if __name__ == "__main__" :
   main(sys.argv)

