#!/bin/bash
rsync -r --exclude-from=.rsyncignore --exclude=.git -e ssh --delete . lisa:~/alfonso/$1

