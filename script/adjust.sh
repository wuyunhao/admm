#!/bin/bash

# include start.sh #######################
source ./script/start.sh

# new init ###############################
L1="1"
L2="10"
ftrl_alpha=0.1
step_size=1
passes=30

#TRAIN_DIR="hdfs://ns1/user/yunhao1/ETL/yunhao_full_20151112_libsvm_train/"
#TEST_DIR="hdfs://ns1/user/yunhao1/ETL/yunhao_full_20151112_libsvm_test/"
TRAIN_DIR="hdfs://ns1/user/zhangwei29/ETL/ftrl_admm_libsvm_train/"
TEST_DIR="hdfs://ns1/user/zhangwei29/ETL/ftrl_admm_libsvm_test/"

EXP=`cat script/exclude_full`
EXP=`echo $EXP|tr ' ' '|'`
INSTANCE=(`hadoop fs -ls ${TEST_DIR}/*/* |awk '$5 > 60'| awk -F'/' '{print $(NF-1)"/"$NF}' | grep -v -E "Found|temporary|$EXP" `)
ITEMS=(`echo ${INSTANCE[@]}|sed 's/\/part-[0-9]*//g'`)
INSTANCE_BACKUP=(${INSTANCE[@]})
ITEMS_BACKUP=(${ITEMS[@]})

function sub_table() {
	for ((fp=0;fp<${num_task};++fp));do
	#for fp in ${INSTANCE};do
    auc=`cat $log | grep "${fp} processor Test AUC" |tail -1|awk '{print $NF}'` 
    echo -n $auc '|' ''>> TABLE_${ITEMS[$fp]}
  done
}

function records() {
  # 表头 #################################
  for fp in ${ITEMS[@]};do
    echo -n '|' '   ' '|' ''> TABLE_${fp}
    for j in $L2;do
      echo -n $j '|' ''>> TABLE_${fp}
    done
    echo >> TABLE_${fp}
  done

  echo > $log
  for i in $L1;do
	# 行信息 ############################# 
	for fp in ${ITEMS[@]};do
      echo -n '|' $i '|' ''>> TABLE_${fp}
	done

    for j in $L2;do
		l_1=$i
        l_2=$j
        ftrl 1>> $log 
		sub_table
    done

	# 换行 ###############################
	for fp in ${ITEMS[@]};do
      echo >> TABLE_${fp} 
	done
  done
}
function tableshow() {
  echo -n '|' '' 'TrainLoss' '|' ''>> $FINALTABLE
  for ((i=0;i<${num_task};++i));do
	echo -n `cat $log | grep " $i processor Train LogLoss" | sed -n "${iter}p"|awk '{print $NF}'` '' '|' ''>> $FINALTABLE
  done
  echo >> $FINALTABLE
  
  echo -n '|' '' 'TestLoss' '|' ''>> $FINALTABLE
  for ((i=0;i<${num_task};++i));do
	echo -n `cat $log | grep " $i processor Test LogLoss" | sed -n "${iter}p"|awk '{print $NF}'` '' '|' ''>> $FINALTABLE
  done
  echo >> $FINALTABLE

  echo -n '|' '' 'TrainAUC' '|' ''>> $FINALTABLE
  for ((i=0;i<${num_task};++i));do
	echo -n `cat $log | grep " $i processor Train AUC" | sed -n "${iter}p"|awk '{print $NF}'` '' '|' ''>> $FINALTABLE
  done
  echo >> $FINALTABLE

  echo -n '|' '' 'TestAUC' '|' ''>> $FINALTABLE
  for ((i=0;i<${num_task};++i));do
	echo -n `cat $log | grep " $i processor Test AUC" |sed -n "${iter}p"| awk '{print $NF}'` '' '|' ''>> $FINALTABLE
  done
  echo >> $FINALTABLE

#  echo -n '|' '' 'TestPV' '|' ''>> $FINALTABLE
#  for ((i=0;i<${num_task};++i));do
#	echo -n `cat $log | grep " $i processor Test PV" |sed -n "${iter}p"| awk '{print $NF}'` '' '|' ''>> $FINALTABLE
#  done
#  echo >> $FINALTABLE
}

function vertical(){
  cols=`awk -F'|' 'END{print NF-1}' $FINALTABLE`
  if [ $cols -lt 3 ];then
	  echo "cols is less than 3, $FINALTABLE is incorrected"
	  exit 1
  fi

  item=`awk -F'|' '{print $2}' $FINALTABLE` 
  item=`echo $item| sed 's/ / | /g; s/^/| =============== |  /; s/$/ |/'`
  echo $item > $VERTICAL 
  
  for ((k=3;k<=$cols;++k));do
    item=`cat $FINALTABLE | awk -F'|' 'BEGIN{j='$k'}{print $j}'` 
    echo $item
	item=`echo $item| sed 's/ / | /g; s/^/| /; s/$/ |/'`
	echo $item >> $VERTICAL
  done
}

function merge_run(){
  NEIBOR=`cat neibor`
  NEIBOR=`echo $NEIBOR | tr ' ' '|'`
  FINALTABLE="FINALTABLE_m"
  cat FINALTABLE_a | grep -E "PDPS|TestAUC|TestPV" > $FINALTABLE
  index=`cat FINALTABLE_f | awk -F'|' 'NR==1{for(i=1;i<=NF;++i)if($i ~ /'$NEIBOR'/ || i==2) print i}'` 
  echo $index
  index=`echo $index | tr ' ' ','`
  echo $index
  cat FINALTABLE_f | cut -d'|' -f$index |sed -n '/TestAUC/p'| sed  's/^/|/; s/$/|/'>> $FINALTABLE
  cat FINALTABLE_f | cut -d'|' -f$index |sed -n '/TestPV/p'| sed  's/^/|/; s/$/|/'>> $FINALTABLE
  VERTICAL="VERTICAL_m"
  vertical
}

function run_records(){
  NEIBOR=`cat neibor`
  NEIBOR=`echo $NEIBOR | sed 's/ /|/g'`
  echo $NEIBOR
  INSTANCE=(`echo ${INSTANCE_BACKUP[@]} |awk '{for(i=1;i<=NF;++i) if($i ~ /'$NEIBOR'/) print $i}'`)
  ITEMS=(`echo ${ITEMS_BACKUP[@]} | awk '{for(i=1;i<=NF;++i) if($i ~ /'$NEIBOR'/) print $i}'`)

  echo ${INSTANCE[@]} | sed 's/ /\n/g' > INSTANCE
  echo ${ITEMS[@]} | sed 's/ /\n/g'> ITEMS
  num_task=${#INSTANCE[@]}
  echo $num_task

  FINALTABLE="FINALTABLE_a"
  VERTICAL="VERTICAL_a" 
  log="log_a"
  
  echo > $log
  for i in $L1;do
    for j in $L2;do
        l_v=$i
	    l_w=$j
        admm 1>> $log
	done
  done
  
  echo -n '|' ''  '  ' '|' '' > $FINALTABLE
  for f in ${ITEMS[@]};do
    echo -n $f '|' '' >> $FINALTABLE
  done
  echo >> $FINALTABLE
  
  for ((iter=1;iter<=`expr 1 '*' $passes`;++iter));do
     tableshow
  done

  echo -n '|' '' 'TestPV' '|' ''>> $FINALTABLE
  for ((i=0;i<${num_task};++i));do
	echo -n `cat $log | grep " $i processor Test PV" |sed -n "1p"| awk '{print $NF}'` '' '|' ''>> $FINALTABLE
  done
  echo >> $FINALTABLE

  vertical
}

function run_evan(){
  ITEM=PDPS000000046019
  COMPARE=`echo ${INSTANCE[@]} | awk '{for(i=1;i<=NF;++i) if($i ~ /'$ITEM'/) print $i}'`

  LOADED_FILE=$OUTPUT_DIR"ftrl_weight_"$COMPARE
  INSTANCE=(`echo ${INSTANCE[@]} | sed -e "s/${ITEM}\/part-[0-9]*//"`)
  ITEMS=(`echo ${ITEMS[@]}| sed "s/${ITEM}//"`)
  num_task=${#INSTANCE[@]} 
  echo $num_task
  echo ${#ITEMS[@]} 
  echo $LOADED_FILE		

  FINALTABLE="FINALTABLE_e"
  VERTICAL="VERTICAL_e"
  log="log_e"
  iter=1

  evan 1> $log

  echo -n '|' ''  '  ' '|' '' > $FINALTABLE
  for f in ${ITEMS[@]};do
    echo -n $f '|' '' >> $FINALTABLE
  done
  echo >> $FINALTABLE

  tableshow
  echo $ITEM > neibor
  vertical
  cat $VERTICAL | awk -F'|' 'NR>1{print $2,$4}'| sort -n -r -k 2|head -10 |awk '{print $1}'>> neibor
}

function run_pred(){
  NEIBOR=`cat neibor`
  NEIBOR=`echo $NEIBOR | sed 's/ /|/g'`
  echo $NEIBOR
  INSTANCE=(`echo ${INSTANCE_BACKUP[@]} |awk '{for(i=1;i<=NF;++i) if($i ~ /'$NEIBOR'/) print $i}'`)
  ITEMS=(`echo ${ITEMS_BACKUP[@]} | awk '{for(i=1;i<=NF;++i) if($i ~ /'$NEIBOR'/) print $i}'`)
  num_task=${#INSTANCE[@]}

  FINALTABLE="FINALTABLE_p"
  VERTICAL="VERTICAL_p"
  log="log_p"
  iter=1

  echo > $log
  L=(0.2 0.25 0.3 0.35 0.4 1)
  for ratio in ${L[@]};do
    pred 1>> $log
  done

  echo -n '|' ''  '  ' '|' '' > $FINALTABLE
  for f in ${ITEMS[@]};do
    echo -n $f '|' '' >> $FINALTABLE
  done
  echo >> $FINALTABLE

  for ((iter=1;iter<=${#L[@]};++iter));do
	tableshow
  done

  echo -n '|' '' 'TestPV' '|' ''>> $FINALTABLE
  for ((i=0;i<${num_task};++i));do
	echo -n `cat $log | grep " $i processor Test PV" |sed -n "1p"| awk '{print $NF}'` '' '|' ''>> $FINALTABLE
  done
  echo >> $FINALTABLE

  vertical
}

function compute(){
  VERTICAL="VERTICAL_m"
  new_auc=`cat $VERTICAL | awk -F'|' 'BEGIN{pv=0;}
							NR>1{
							pv+=$(NF-1)*$(NF-1);
							for(i=3;i<NF-1;++i)
							  auc[i]=auc[i]+($i)*$(NF-1)*$(NF-1);
							}
							END{
							for(i=3;i<NF-1;++i)
								printf("%f\n", auc[i]/pv);
							}'`
  echo $new_auc
  new_auc=`echo $new_auc | sed -e 's/ / | /g' -e 's/^/| =============== | /' -e 's/$/ |   |/'` 
  echo $new_auc >> $VERTICAL

  promote=`cat $VERTICAL|tail -1| awk -F'|' '{for(i=3;i<NF-1;++i) printf("%.4f ", $i - $(NF-2));}'`
  promote=`echo $promote | sed -e 's/ / | /g' -e 's/^/| =============== | /' -e 's/$/ |   |/'` 
  echo $promote >> $VERTICAL
}
run_records
