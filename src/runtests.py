from cvxpy import *
import numpy as np
import threading
import time
import os, sys, inspect
import pandas as pd


# check runtime:
startall = time.time()

# Class for threading a single problem through multiple solver configurations.
class ProblemThread(threading.Thread):
	"""A cvxpy thread-safe problem solving object."""
	def __init__(self, probID, prob, configs):
		threading.Thread.__init__(self)
		self.probID = probID
		self.prob = prob
		self.configs = configs

	def run(self):
		for configID in configs:
			config = configs[configID]
			# output = {}
			# Default values:
			runtime = "-"
			status = "-"
			opt_val = "-"

			try:
				start = time.time() # Time the solve
				self.prob.solve(**config)
				runtime = time.time() - start
				status = self.prob.status
				opt_val = self.prob.value
			except:
				# Configuration could not solve the given problem
				print "failure in solving."
			# output["status"] = status
			# output["opt_val"] = opt_val
			# output["time"] = runtime
			outputLock.acquire()
			problemOutputs.loc[:, self.probID, configID] = [status, runtime, opt_val]
			outputLock.release()
	# def writeResults(self, configID, outputDict):
	# 	"""Modify to change what results we record."""
	# 	resultsList = [
	# 		self.prob.status,
	# 		self.prob.value,
	# 		self.time
	# 	]
	# 	problemOutputs[(self.probID, configID)] = resultsList # Thread-safe via GIL

# Lock for the outputDict:
outputLock = threading.Lock()

# Read in problems
# cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
# if cmd_folder not in sys.path:
#     sys.path.insert(0, cmd_folder)

# use this if you want to include problems from a subfolder
cmd_folder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"problems")))
if cmd_folder not in sys.path:
	sys.path.insert(0, cmd_folder)


# Create solver configurations
configs ={solver : {"solver": solver} for solver in ['CVXOPT', 'ECOS_BB', 'SCS', 'ECOS']}
problemDict = {}


for dirname, dirnames, filenames in os.walk(cmd_folder):
	# print path to all subdirectories first.
	# for subdirname in dirnames:
	# 	print(os.path.join(dirname, subdirname))
	# print all filenames.
	for filename in filenames:
		# print filename
		if filename[-3:] == ".py" and filename != "__init__.py":
			problemID = filename[0:-3]
			# Set up thread and lock for this problem
			problemDict[problemID] = ProblemThread(problemID, __import__(problemID).prob, configs)
			# for configID in configs:
			# 	problemOutputs[(problemID, configID)] = {} # Populate in advance.
	# Advanced usage:
	# editing the 'dirnames' list will stop os.walk() from recursing ipnto there.
	# if '.git' in dirnames:
	# 	# don't go into any .git directories.
	# 	dirnames.remove('.git')
problemOutputs = pd.Panel(	items = ["status", "time","opt_val"], 
							major_axis = [problemID for problemID in problemDict], 
							minor_axis = [config for config in configs])


# Run every solver configuration against every problem and save the results

# Set up threads and locks:
# for problemID in problemDict:
# 	problemLocks[problemID] = threading.Lock()
# 	problemDict[problemID] = ProblemThread(problemID, problemDict[problemID])

for problemID in problemDict:
	# Set (config, problem) off on its own thread
	if __name__ == "__main__":
		# try:
		problemDict[problemID].start()
		# except:
			# print "Error: unable to start thread."

	# problemDict[problemID].solve(**(configs[configID]))
	# print problem.status
	# print problem.value

# Wait for threads to finish:
for problemID in problemDict:
	problemDict[problemID].join()

# Display results

# for key in problemOutp:
# 	print key, ":", problemOutput[key]

print problemOutputs.to_frame()


print "Runtime:",str(time.time() - startall)
