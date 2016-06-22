from cvxpy import *
import numpy as np
import threading
import time


# Class for threading a single problem through multiple solver configurations.
class ProblemThread(threading.Thread):
	"""A cvxpy thread-safe problem solving object."""
	def __init__(self, probID, prob):
		threading.Thread.__init__(self)
		self.probID = probID
		self.prob = prob
		self.time = -1

	def run(self, configID, config, outputDict):
		# Apply lock on the prob object
		probLock.acquire()
		start = time.time() # Time the solve
		self.prob.solve(**config)
		self.time = time.time() - start
		self.writeResults(configID, outputDict)
		# Free lock to release next thread
		probLock.release()

	def writeResults(self, configID, outputDict):
		resultsList = [
			self.prob.status,
			self.prob.value,
			self.time
		]
		outputDict[(self.probID, configID)] = resultsList # Thread-safe via GIL

# Lock for the problem thread
probLock = threading.Lock()

# Read in problems
import os, sys, inspect
# cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
# if cmd_folder not in sys.path:
#     sys.path.insert(0, cmd_folder)

# use this if you want to include problems from a subfolder
cmd_folder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],"problems")))
if cmd_folder not in sys.path:
	sys.path.insert(0, cmd_folder)

print sys.path

print cmd_folder
problemDict = {}
for dirname, dirnames, filenames in os.walk(cmd_folder):
	# print path to all subdirectories first.
	# for subdirname in dirnames:
	# 	print(os.path.join(dirname, subdirname))
	# print all filenames.
	for filename in filenames:
		# print filename
		if filename[-3:] == ".py" and filename != "__init__.py":
			# Dynamic imports:
			problemID = filename[0:-3]
			print problemID
			problemDict[problemID] = __import__(problemID).prob # prob is the variable name
	# Advanced usage:
	# editing the 'dirnames' list will stop os.walk() from recursing into there.
	# if '.git' in dirnames:
	# 	# don't go into any .git directories.
	# 	dirnames.remove('.git')

# Create solver configurations
# print len(problems)

configs ={solver : {"solver": solver} for solver in ['ECOS_BB', 'SCS', 'ECOS']}

# Run every solver configuration against every problem and save the results
outputDict = {}
for problemID in problemDict:
	probThread = ProblemThread(problemID, problemDict[problemID])
	for configID in configs:
		# Set (config, problem) off on its own thread
		if __name__ == "__main__":
			# try:
				probThread.run(configID, configs[configID], outputDict)
			# except:
				# print "Error: unable to start thread."

		# problem.solve(**config)
		# print problem.status
		# print problem.value


for key in outputDict:
	print key, ":", outputDict[key]


# # Test:


# m = 20
# n = 10
# np.random.seed(1)

# x = Variable(n)
# A = np.random.rand(m, n)
# b = np.random.rand(m)

# objective = Minimize(sum_squares(A*x - b))
# prob = Problem(objective)

# for config in configs:
# 	prob.solve(**config)
# 	print "status:", prob.status
# 	print "optimal value", prob.value

