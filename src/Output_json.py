# coding:utf-8
import sys
import os
import sqlite3
from helpers import *

os.chdir('..')

# args are: congress_db, networks_db, output_dir, time_var, start, end, location:
congressdb = sys.argv[1]
networksdb = sys.argv[2]
output_dir = sys.argv[3]
time_var = sys.argv[4]
start = int(sys.argv[5])
end = int(sys.argv[6])
location = sys.argv[7]
threshold = float(sys.argv[8])

networks_con = sqlite3.connect(os.path.join('data',networksdb))
congress_con = sqlite3.connect(os.path.join('data',congressdb))

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

write_json(networks_con=networks_con,congress_con=congress_con,
            output_dir=output_dir,time_var=time_var,
            start=start,end=end,location=location,agreement_threshold=threshold)

