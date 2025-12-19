#!/bin/bash

export SRCFILE=$1  # filename in data/processed
export NUM_MUT=$2  # number of mutations to screen
export NUM_ITER=$3  # number of iterations to run


# Loop through the specified number of iterations
i=0
while [ $i -lt $NUM_ITER ]
do
    if [ $i -eq 0 ]; then
        CURRENT_FILE=$SRCFILE
    else
        CURRENT_FILE="${SRCFILE}_mut${i}"
    fi
    echo "Iteration $((i+1))/$NUM_ITER, Current file: $CURRENT_FILE"

    # xlsx to npz
    python -m preprocess.process_mut --srcfilename=$CURRENT_FILE.xlsx

    # extract t5 feature and will save output in data/FEATTYPE/SRCFILE_NAME
    python -m preprocess.extract_feat --feature_type=t5 --srcfilename=$CURRENT_FILE.npz

    # Run the mutation screen and save output in data/processed
    python LLPSense_standalone.py data.mut_file=data/processed/$CURRENT_FILE.npz standalone.num_mut=$NUM_MUT standalone.exp_task=mut_screen standalone.screen=temp expname=$CURRENT_FILE standalone.mut_direction=positive

    echo "Iteration $((i+1)) completed successfully."
    
    # Increment the counter
    i=$((i + 1))
done


# example:
# sh scripts/mut_screen.sh UBQLN1 1 10