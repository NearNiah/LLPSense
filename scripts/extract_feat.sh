
export FEATTYPE=$1  # model config name, t5 or t5_seq or eng
export SRCFILE=$2  # filename in data/processed

# extract feature and will save output in data/FEATTYPE/SRCFILE_NAME
python -m preprocess.extract_feat --feature_type=$FEATTYPE --srcfilename=$SRCFILE

# Usage:
# In LLPSense root directory, run:
# .npz file should be in data/processed
# sh scripts/extract_feat.sh t5 LLPSDB.npz