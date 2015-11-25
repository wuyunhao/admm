# distr_admm
Distributed ADMM for linear models.

# Algo

# System

# Build & Run

##### 1 Clone git repo
```
git clone http://10.210.228.76/opticlick/admm.git
```

##### 2 Bootstrap the project dev environment
```
./bootstrap
```
It will download/checkout all the third-party libraries and build them.

##### 3 Build
```
make && make check
```

##### 4 Run locally
```
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native:$JAVA_HOME/jre/lib/amd64:$JAVA_HOME/jre/lib/amd64/server DMLC_ROLE=worker ./tracker/dmlc_local.py -n 3  --log-level DEBUG  ./admm
```

##### 5 Run on YARN

Example:
```
tracker/dmlc_yarn.py -n 6 --vcores 1 --log-level DEBUG -q root.megatron.baigang --jobname admm_lr --tempdir ns1/user/baigang/temp -mem 512   ./admm
```

##### 6 Run via script

The configuration of admm is set in script/start.sh and script/adjust.sh, including paths of training data and testing data, regulation coefficients, output directory and so on. After specified the paramaters in these .sh scripts, admm can be run in below commands:
Run in YARN mode 
```
sh script/adjust.sh 1
```

Run in local mode
```
sh script/adjust.sh 0
```

There would be a log file named log_a is generated in the current directory


