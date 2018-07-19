import os
import sys
import sqlite3
import re

from lsst.sims.featureScheduler.coldstart.config import *


"""The database class is responsible for configuring all database related tasks.
This includes connection to "*.db" files and exposing it's database related 
resources to outside classes, for example a cursor. 
"""
class database:

	# create a list of all the database files
	files = [f for f in os.listdir(DB_DIRECTORY)
		if os.path.isfile(os.path.join(DB_DIRECTORY, f)) and ".db" in f]

	# create a list of all the log files
	logFiles = [f for f in os.listdir(LOG_DIRECTORY)
		if os.path.isfile(os.path.join(LOG_DIRECTORY, f)) and ".log" in f]

	# Create a string friendly list of files found in DB_DIRECTORY
	allFiles = ""
	for each in files:
		if len(allFiles.split("\n")[-1] + each) > 80:
			allFiles += "\n"

		allFiles += each + ", "

	# Some public instance variables that are useful for definitions
	path = ""
	dbNumber = -1

	"""Creates a cursor to a database to be used within the other scripts. Also
	responsible for printing out useful information to the user along with 
	obtaining the file that the user wishes to analyse
	"""
	def connect(self,filePath = None):

		if filePath :
			databaseFile=filePath

		else:

			print("=" * 80)
			print("Found [" + str(len(self.files)) + "] databases inside of '" + DB_DIRECTORY + "'")
			print("-" * 80)
			print('%5s' % self.allFiles)
			print(("=" * 80) +"\n")

			databaseFile = str(input('Please type the name of the database you use for Cold Start: '))
			
		while True:

			if databaseFile in self.files:
				break

			else:
				print("\n" + ("-"* 80))
				print("!!! Could not find '" + databaseFile + "' inside of '" + DB_DIRECTORY + "' !!!")
				print(("-" *80) + "\n")

				databaseFile = str(input('try again: '))

		self.path = DB_DIRECTORY + databaseFile
		# the number is the value between the last _ and . 
		self.dbNumber = databaseFile.split('_')[-1].split('.')[0]

		conn = sqlite3.connect(self.path)
		curs = conn.cursor()
		return curs