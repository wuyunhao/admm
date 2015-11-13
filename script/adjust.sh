#!/bin/bash

source ./script/start.sh

L1="0.05"
L2="0.2 0.4"

function write() {
  echo -n '   ' '' > TABLE
  for j in $L2;do
    echo -n $j '' >> TABLE
  done
  echo >> TABLE
  
  for i in $L1;do
    echo -n $i ''>> TABLE
    for j in $L2;do
      l_1=$i
      l_2=$j
      ftrl 1> log1 
      auc=`cat log | grep "Test AUC" |tail -1|awk '{print $NF}'` 
      echo -n $auc '' >> TABLE
    done
    echo >> TABLE 
  done
}
evan
