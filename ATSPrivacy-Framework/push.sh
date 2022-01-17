#!/bin/bash
rsync -r --exclude-from=.rsyncignore --exclude=.git -e ssh --delete . lisa:$1

