GPU=0
MAX=0
while true; do
  USED=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$USED" -gt "$MAX" ]; then MAX=$USED; fi
  echo "used=${USED} MiB  max=${MAX} MiB"
  sleep 0.2

done
