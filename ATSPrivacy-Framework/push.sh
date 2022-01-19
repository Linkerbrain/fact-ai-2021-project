#!/bin/bash
if [ "$#" -eq  "0" ]
  then
    echo "No arguments supplied"
else
    rsync -r --exclude-from=.rsyncignore --exclude=.git -e ssh --delete . lisa:~/$1
fi


