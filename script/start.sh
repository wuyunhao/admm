#!/bin/bash

# init #########################
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib:$HADOOP_HOME/lib/native:$JAVA_HOME/lib/amd64:$JAVA_HOME/lib/amd64/server
DMLC_ROLE=worker
TRAIN_DIR="data/"
TEST_DIR="data/"
INSTANCE="00001 00002"

# ftrl ##########################
l_1=0.05
l_2=60

# admm ##########################
l_w=4
l_v=0.01
step_size=0.01

# common ########################
dim=4771
ftrl_alpha=0.04
ftrl_beta=1
passes=8


function ftrl(){
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib:$HADOOP_HOME/lib/native:$JAVA_HOME/lib/amd64:$JAVA_HOME/lib/amd64/server DMLC_ROLE=worker \
    ./tracker/dmlc_local.py -n 1  --log-level DEBUG  ./ftrl \
    $l_1 $l_2 $ftrl_alpha $ftrl_beta $dim $passes \
    $TRAIN_DIR $TEST_DIR $INSTANCE
}

function admm(){
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib:$HADOOP_HOME/lib/native:$JAVA_HOME/lib/amd64:$JAVA_HOME/lib/amd64/server DMLC_ROLE=worker \
    ./tracker/dmlc_local.py -n 1  --log-level DEBUG ./admm \
    $l_w $l_v $step_size $ftrl_alpha $dim $passes \
    $TRAIN_DIR $TEST_DIR $INSTANCE
}

