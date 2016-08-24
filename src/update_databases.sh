#!/bin/bash

py=$(cat ../config/PYTHONPATH.txt) #'/home/matt/anaconda3/bin/python3.5'

TIMEVAR="month"
LAG=8
CONGRESS_DB="congress.db"
NETWORKS_DB="congress_networks.db"

# pull any new data from govtrack
bash sync_with_govtrack.sh

# update the master db.  Optional argument is output db name
$py Import_govtrack_to_sql.py $CONGRESS_DB

# make indices for fast queries
$py Create_indices.py $CONGRESS_DB

# compute senator-senator agreement and senator temporal data
# for node filtering and upload to the network db
# Arg 1 is input db name, arg 2 is output db name, trailing args are time variables
# on which to aggregate votes with lags
# e.g " year 0 month 4 " would aggregate in 1-year and 5-month timeslices
$py Extract_networks_to_sql.py $CONGRESS_DB $NETWORKS_DB $TIMEVAR $LAG

# make indices for fast queries
$py Create_indices.py $NETWORKS_DB

# compute and upload network x-y embeddings
$py Upload_embeddings_to_sql.py $CONGRESS_DB $NETWORKS_DB

