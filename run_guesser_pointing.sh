#!/bin/bash

if [ -z ${config+x} ]; then
	echo "no config file was provided";
        exit 1
fi

upload_speed=50000

SECONDS=0 ;
DATA_DIR_IN=/home/fstrub/data/guesswhat_data

if [[ $* == *--no_rsync* ]]; then
	echo "Use data on shared drive"
	DATA_DIR_OUT=$DATA_DIR_IN
else
	echo "Use Data on local drive (require rsync fro share drive)"
	DATA_DIR_OUT=$SLURM_TMPDIR/guesswhat_data
	mkdir -p $DATA_DIR_OUT
	rsync -ru --bwlimit=$upload_speed --progress $DATA_DIR_IN/conv3_crop_h5 $DATA_DIR_OUT/
	rsync -ru --bwlimit=$upload_speed --progress $DATA_DIR_IN/conv3_img_h5 $DATA_DIR_OUT/
	rsync -ru --bwlimit=$upload_speed --progress $DATA_DIR_IN/guesswhat.test.jsonl.gz $DATA_DIR_OUT/
	rsync -ru --bwlimit=$upload_speed --progress $DATA_DIR_IN/guesswhat.train.jsonl.gz $DATA_DIR_OUT/
	rsync -ru --bwlimit=$upload_speed --progress $DATA_DIR_IN/guesswhat.valid.jsonl.gz $DATA_DIR_OUT/
	rsync -ru --bwlimit=$upload_speed --progress $DATA_DIR_IN/dict.json $DATA_DIR_OUT/
	rsync -ru --bwlimit=$upload_speed --progress $DATA_DIR_IN/glove_dict.pkl $DATA_DIR_OUT/
	echo "Time to copy data: " $SECONDS
fi

cd /home/fstrub/secret_s
export PYTHONPATH="."
cd /home/fstrub/secret_s/guesswhat_s/src
python3 guesswhat/train/train_guesser.py -data_dir $DATA_DIR_OUT \
	-exp_dir /home/fstrub/secret_s/guesswhat_s/out/guesser \
	-img_dir $DATA_DIR_OUT/conv3_img_h5 \
	-glove_file $DATA_DIR_OUT/glove_dict.pkl \
	-config /home/fstrub/secret_s/guesswhat_s/config/guesser/pointing/${config} \
	-dict_file $DATA_DIR_IN/dict.json \
	-gpu_ratio 0.95 \
	-no_thread 4 \
    -pointing_task true