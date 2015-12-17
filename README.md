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
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native:$JAVA_HOME/jre/lib/amd64:$JAVA_HOME/jre/lib/amd64/server DMLC_ROLE=worker ./tracker/dmlc_local.py -n 3  --log-level DEBUG  ./admm admm.conf
```

##### 5 Run on YARN

Example:
```
tracker/dmlc_yarn.py -n 6 --vcores 1 --log-level DEBUG -q root.megatron.baigang --jobname admm_lr --tempdir ns1/user/baigang/temp -mem 512   ./admm admm.conf
```

