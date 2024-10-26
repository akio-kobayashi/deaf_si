#!/bin/sh

# uidは自分の計算機の/etc/passwdを見るなどして変更する
uid=1000
image=peft

docker run --shm-size 16394m \
       --rm -it -v /tmp:/tmp -v /mnt/:/mnt -v /home:/home \
       -v /media/:/media -v /etc/group:/etc/group:ro \
       -v /export1/:/export1 -v /export2:/export2 \
       -v /etc/passwd:/etc/passwd:ro -u ${uid}:${uid} --gpus all \
       $image bash
