#encoding:utf-8

import os
import sys
from helpers import *

# initialize the tables
os.chdir('..')

# initialize the database
db = sys.argv[1]#'congress.db'
db_name = os.path.splitext(db)[0]
schema = read_schema(os.path.join('config',db_name + '_schema'),extensions=['.yml','.json'],legal_types=sqlite_types)

os.chdir('data')

# connect
con = sqlite3.connect(db)
cur = con.cursor()

# create indices for fast queries
for table, table_info in schema.items():
    if 'indices' in table_info:
        print("Creating indices for table {}".format(table))
        create_index(con,**table_info)

# close the connection
cur.close()
con.close()

exit()

