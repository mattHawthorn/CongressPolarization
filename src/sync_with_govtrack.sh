#!/bin/bash

cd ..

if [ ! -d data ]; then 
mkdir data
fi

cd data

# absence of trailing slash on the source dir indicates creating a dir in the cwd called "congress"
rsync -avz --delete --delete-excluded --exclude **/text-versions/ --exclude ***.json --exclude **/h*/ \
		govtrack.us::govtrackdata/congress .
		
rsync -avz --delete --delete-excluded --exclude **/text-versions/ \
		govtrack.us::govtrackdata/congress-legislators .

