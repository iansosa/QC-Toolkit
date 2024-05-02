#! /bin/bash

echo "Would you like to propagate using propagation.txt? (y/N)"
read prop
if [ $prop = "y" ]; then
  rsync -avzu --progress -e 'ssh -p 8022' * isosa@access-aion.uni.lu:EB/
  declare -a myArray
  file=propagation.txt
  myArray=(`cat "$file"`)
  length=${#myArray[@]}
  echo "names to be created: ${myArray[*]}"
  for (( i=0; i<${length}; i++ ));
  do
    echo "index $i/$length"
    echo "creating: EB_${myArray[$i]}"
    ssh -p 8022 isosa@access-aion.uni.lu cp -rd EB "EB_${myArray[$i]}"
  done
  ssh -p 8022 isosa@access-aion.uni.lu rm -rd EB
else
  echo "How many copies of the same folder?"
  read number
  rsync -avzu --progress -e 'ssh -p 8022' * isosa@access-aion.uni.lu:EB_1/

  i=1
  while (( i++ < number )); do
    printf "copying $i ..."
    ssh -p 8022 isosa@access-aion.uni.lu cp -rd EB_1 "EB_$i"
  done
fi