
export SRCFILE=$1  # filename in data/processed

python -m preprocess.process_db --srcfilename=$SRCFILE

# Usage:
# In LLPSense root directory, run:
# .xlsx file should be in data/raw, output will be saved in data/processed
# sh scripts/process_db.sh LLPSDB.xlsx