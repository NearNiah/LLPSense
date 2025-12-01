
export SRCFILE=$1  # filename in data/processed

# extract feature and will save output in data/FEATTYPE/SRCFILE_NAME
python -m preprocess.random_split --srcfilename=$SRCFILE

# Usage:
# In LLPSense root directory, run:
# .npz file should be in data/processed
# sh scripts/random_split.sh LLPSDB.npz