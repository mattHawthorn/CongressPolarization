#!/bin/bash

py=$(cat ../config/PYTHONPATH.txt) #'/home/matt/anaconda3/bin/python3.5'

# start year and month
month1=$1
year1=$2
# end year and month
month2=$3
year2=$4
# threshold of agreement to trim down the edge count
threshold=$5

start=$(($year1*12+$month1))
end=$(($year2*12+$month2))

# args are: congress_db, networks_db, output_dir, time_var, start, end, location, agreement_threshold:
$py Output_json.py congress.db congress_networks.db app/json month $start $end senate $threshold
