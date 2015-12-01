#!/bin/bash

mode=$1

# yarn init ##########################################
HADOOP=`which hadoop`
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native:$JAVA_HOME/jre/lib/amd64:$JAVA_HOME/jre/lib/amd64/server
DMLC_ROLE=worker 
TRAIN_DIR="hdfs://ns1/user/yunhao1/admm/traindata/"
TEST_DIR="hdfs://ns1/user/yunhao1/admm/testdata/"
OUTPUT_DIR="hdfs://ns1/user/yunhao1/dir-weights/"
INPUT_DIR="hdfs://ns1/user/yunhao1/dir-weights/"
#TRAIN_DIR="data/"
#TEST_DIR="data/"
INSTANCE=''
num_task=''

#common configure ####################################
dim=2000000
ftrl_alpha=0.01
ftrl_beta=1
passes=4
ratio=0.4

INSTANCE=(PDPS000000005427)
num_task=${#INSTANCE[@]}
#echo $num_task
#echo ${INSTANCE[@]}

#admm configure ######################################
l_w=4
l_v=0.01
step_size=0.01

#ftrl configure ######################################
l_1=0.4
l_2=2.5

# ftrl ###############################################
function ftrl(){
  if [ $mode -eq 0 ];then
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native:$JAVA_HOME/jre/lib/amd64:$JAVA_HOME/jre/lib/amd64/server  DMLC_ROLE=worker \
	./tracker/dmlc_local.py -n $num_task  --log-level DEBUG  ./ftrl \
	  $l_1 $l_2 $ftrl_alpha $ftrl_beta $dim $passes \
	  $TRAIN_DIR $TEST_DIR $OUTPUT_DIR \
	  ${INSTANCE[@]}
  else
	./tracker/dmlc_yarn.py -n $num_task --vcores 1 --log-level DEBUG -q root.megatron.yunhao1 --jobname admm_lr --tempdir ns1/user/yunhao1/temp -mem 5120 ./ftrl \
	  $l_1 $l_2 $ftrl_alpha $ftrl_beta $dim $passes \
	  $TRAIN_DIR $TEST_DIR $OUTPUT_DIR \
	  ${INSTANCE[@]}
  fi
} 

# admm ###############################################
function admm(){
  if [  $mode -eq 0 ];then
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native:$JAVA_HOME/jre/lib/amd64:$JAVA_HOME/jre/lib/amd64/server \
    DMLC_ROLE=worker \
	./tracker/dmlc_local.py -n $num_task --log-level DEBUG  ./admm \
	  $l_w $l_v $step_size $ftrl_alpha $dim $passes \
	  $TRAIN_DIR $TEST_DIR $OUTPUT_DIR \
	  ${INSTANCE[@]}
  else
	./tracker/dmlc_yarn.py -n $num_task --vcores 1 --log-level DEBUG -q root.megatron.yunhao1 --jobname admm_lr --tempdir ns1/user/yunhao1/temp -mem 5120 ./admm \
	  $l_w $l_v $step_size $ftrl_alpha $dim $passes \
	  $TRAIN_DIR $TEST_DIR $OUTPUT_DIR \
	  ${INSTANCE[@]}
  fi
}

# eval ###############################################
function evan(){
  if [ $mode -eq 0 ];then
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native:$JAVA_HOME/jre/lib/amd64:$JAVA_HOME/jre/lib/amd64/server \
    DMLC_ROLE=worker \
	./tracker/dmlc_local.py -n ${#INSTANCE[@]} --log-level DEBUG  ./evan \
	  $LOADED_FILE $TRAIN_DIR $TEST_DIR ${INSTANCE[@]}
  else
	./tracker/dmlc_yarn.py -n ${#INSTANCE[@]} --vcores 1 --log-level DEBUG -q root.megatron.yunhao1 --jobname admm_lr --tempdir ns1/user/yunhao1/temp -mem 4086 ./evan \
	  $LOADED_FILE $TRAIN_DIR $TEST_DIR ${INSTANCE[@]}
  fi
}

# pred ###############################################
function pred(){
  if [ $mode -eq 0 ];then
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native:$JAVA_HOME/jre/lib/amd64:$JAVA_HOME/jre/lib/amd64/server \
    DMLC_ROLE=worker \
	./tracker/dmlc_local.py -n ${#INSTANCE[@]} --log-level DEBUG  ./pred \
	  $dim $ratio $TRAIN_DIR $TEST_DIR $INPUT_DIR \
	  ${INSTANCE[@]}
  else
	./tracker/dmlc_yarn.py -n ${#INSTANCE[@]} --vcores 1 --log-level DEBUG -q root.megatron.yunhao1 --jobname admm_lr --tempdir ns1/user/yunhao1/temp -mem 4086 ./pred \
	  $dim $ratio $TRAIN_DIR $TEST_DIR $INPUT_DIR \
	  ${INSTANCE[@]}
  fi
}

# sgd ###############################################
function sgd(){
  if [ $mode -eq 0 ];then
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native:$JAVA_HOME/jre/lib/amd64:$JAVA_HOME/jre/lib/amd64/server  DMLC_ROLE=worker \
	./tracker/dmlc_local.py -n $num_task  --log-level DEBUG  ./sgd \
	  $l_1 $l_2 $ftrl_alpha $ftrl_beta $dim $passes \
	  $TRAIN_DIR $TEST_DIR $OUTPUT_DIR \
	  ${INSTANCE[@]}
  else
	./tracker/dmlc_yarn.py -n $num_task --vcores 1 --log-level DEBUG -q root.megatron.yunhao1 --jobname admm_lr --tempdir ns1/user/yunhao1/temp -mem 5120 ./sgd \
	  $l_1 $l_2 $ftrl_alpha $ftrl_beta $dim $passes \
	  $TRAIN_DIR $TEST_DIR $OUTPUT_DIR \
	  ${INSTANCE[@]}
  fi
} 
