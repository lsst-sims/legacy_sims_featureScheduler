# Absolute folder path that contains all your databases
DB_DIRECTORY = "/home/opsim/run_local/output/"

# Absolute folder path that contains all your logs
LOG_DIRECTORY = "/home/opsim/run_local/"

# Folder path which reports are written into, optional
REPORT_DIRECTORY = "reports/"

# Python 2 & 3 compatible function binding
try:
   input = raw_input
except NameError:
   pass
