#!/bin/bash

py=$(cat ../config/PYTHONPATH.txt) #'/home/matt/anaconda3/bin/python3.5'

# pull any new data from govtrack
bash sync_with_govtrack.sh

# update the master db.  Optional argument is output db name
$py Import_govtrack_to_sql.py congress.db

# make indices for fast queries
$py Create_indices.py congress.db

# compute senator-senator agreement and senator temporal data
# for node filtering and upload to the network db
# Arg 1 is input db name, arg 2 is output db name, trailing args are time variables
# on which to aggregate votes with lags
# e.g " year 0 month 4 " would aggregate in 1-year and 5-month timeslices
$py Extract_networks_to_sql.py congress.db congress_networks.db month 6

# make indices for fast queries
$py Create_indices.py congress_networks.db

# compute and upload network x-y embeddings
$py Upload_embeddings_to_sql.py congress.db congress_networks.db

