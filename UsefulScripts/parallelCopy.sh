SOURCEDIR="/examplevol/users/saibi/data/nrbeta"
TARGETDIR="/mshvol/users/saibi/data"
MAX_PARALLEL=4
nroffiles=$(ls "$SOURCEDIR" | wc -w)
setsize=$(( nroffiles/MAX_PARALLEL + 1 ))
ls -1 "$SOURCEDIR"/* | xargs -n "$setsize" | while read workset; do
  cp -prnv "$workset" "$TARGETDIR" &
done
wait
