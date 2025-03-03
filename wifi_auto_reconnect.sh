#!/bin/bash

CONNECTION=$1 #接続名
INTERVAL=$2 # チェック周期[sec]
WAIT=$3 # 起動時待ち時間[sec]

echo "connection=$CONNECTION"
echo "INTERVAL=$INTERVAL sec"

# 起動時待ち時間
# すぐに動かしたくないので適宜入れる
echo "waiting $WAIT sec..."
sleep $WAIT

# 繰り返し
while true
do

    # IPアドレスを取得
    x=`nmcli c show $CONNECTION | grep IP4.ADDRESS`

    if [ "$x" == "" ]; then
        # IPアドレスが空なら接続されていないので接続
        nmcli c up $CONNECTION
    fi

    # 待つ
    sleep $INTERVAL

done
