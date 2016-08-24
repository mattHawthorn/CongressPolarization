#!/bin/bash

py=$(cat ../config/PYTHONPATH.txt) #'/home/matt/anaconda3/bin/python3.5'

CONGRESS_DB="congress.db"
NETWORKS_DB="congress_networks.db"
OUTDIR="app/json"
TIMEVAR="month"
START_YR=1950
START_MO=1
END_YR=2016
END_MO=6
START=$(($START_MO + 12*$START_YR))
END=$(($END_MO + 12*$END_YR))
LOCATIO="senate"

# args are: congress_db, networks_db, output_dir, time_var, start, end, location:
py Output_json.py $CONGRESS_DB $NETWORKS_DB $OUTDIR $TIMEVAR $START $END $LOCATION
