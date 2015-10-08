#!/bin/bash

#ftrl
best_lamda1=1
best_lamda2=1
lamda1=0.03
lamda2=0.02
alpha=0.34
beta=1
#admm
best_lamda_w=1
best_lamda_v=1
best_p=1
#common
auc=0.0
path=data/
file=unlabeled.review #agaricus.txt

function ftrl(){
   li="0.00001 0.0001 0.001 0.01 0.1"
   lj="0.2 0.4 0.6 0.8"
   if [ -e "ftrl_op.txt" ];then
       rm "ftrl_op.txt"
   fi
   
   for niter in 10 ;do
       for alpha in 0.34 ;do
          LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib:$HADOOP_HOME/lib/native:$JAVA_HOME/lib/amd64:$JAVA_HOME/lib/amd64/server DMLC_ROLE=worker ./tracker/dmlc_local.py -n 1  --log-level DEBUG  ./ftrl  $lamda1 $lamda2 $alpha $beta $niter 4771 $path 00001
          #tmp=`./script/auc.py "${path}ftrl_auc"` 
          #echo $lamda2 $alpha $tmp >> "ftrl_op.txt"
          #if [ `expr $auc \< $tmp` -eq 1 ];then
          #    best_lamda1=$lamda1
          #    best_lamda2=$lamda2
          #    auc=$tmp
          #fi
       done
   done
   #echo $best_lamda1  $best_lamda2  $auc
}

function admm(){
   li="1"
   lj="0.03"
   if [ -e "admm_op.txt" ];then
        rm "admm_op.txt"
   fi
   
   for p in $lj ;do
     for lamda_w in 1 ;do
       for lamda_v in $li;do
         LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib:$HADOOP_HOME/lib/native:$JAVA_HOME/lib/amd64:$JAVA_HOME/lib/amd64/server DMLC_ROLE=worker ./tracker/dmlc_local.py -n 1  --log-level DEBUG  ./admm $lamda_w $lamda_v $p $alpha 4771 $path 4 
            #tmp=`./script/auc.py "${path}admm_auc_1"`
            #echo $lamda_w $lamda_v $p $tmp >> "admm_op.txt"
            #if [ `expr $auc \< $tmp` -eq 1 ];then
            #    best_lamda_w=$lamda_w
            #    best_lamda_v=$lamda_v
            #    best_p=$p
            #    auc=$tmp
            #fi
         done
     done
   done
   #echo $best_lamda_w $best_lamda_v $best_p $auc
}

if [ $1 -eq 0 ];then
    ftrl
elif [ $1 -eq 1 ];then
    admm
else
    rm data/admm_auc_1 admm_op.txt ftrl_op.txt data/ftrl_auc
fi
