URL=http://gelsight.csail.mit.edu/wedge/models/weights.h5
FILENAME=./models/weights.h5
wget $URL -O $FILENAME

URL=http://gelsight.csail.mit.edu/wedge/models/weights_generic.h5
FILENAME=./models/weights_generic.h5
wget $URL -O $FILENAME

### Backup weights.h5
### URL=https://drive.google.com/file/d/1K5VMNCY0Ycqtz2O3qdopdAk4rzDW3ZuW/view?usp=sharing
# FILEID=1K5VMNCY0Ycqtz2O3qdopdAk4rzDW3ZuW
# CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
# URL="https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$FILEID"
# FILENAME=./models/weights.h5
# wget --load-cookies /tmp/cookies.txt $URL -O $FILENAME && rm -rf /tmp/cookies.txt


### Backup weights_generic.h5
### URL=https://drive.google.com/file/d/1oqSc0by61_-6HVbCa8d2gZw0OA4xcfEU/view?usp=sharing
# FILEID=1oqSc0by61_-6HVbCa8d2gZw0OA4xcfEU
# FILENAME=./models/weights_generic.h5
# URL="https://docs.google.com/uc?export=download&id=$FILEID"
# wget --no-check-certificate $URL -O $FILENAME