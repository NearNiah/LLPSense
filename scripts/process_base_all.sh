
# xlsx to npz
python -m preprocess.process_db --srcfilename=LLPSDB.xlsx

python -m preprocess.process_db --srcfilename=PSPire_train.xlsx

python -m preprocess.process_db --srcfilename=PSPire_test_ID.xlsx

python -m preprocess.process_db --srcfilename=PSPire_test_noID.xlsx


# extract t5 feature and will save output in data/FEATTYPE/SRCFILE_NAME
python -m preprocess.extract_feat --feature_type=t5 --srcfilename=LLPSDB.npz

python -m preprocess.extract_feat --feature_type=t5 --srcfilename=PSPire_train.npz

python -m preprocess.extract_feat --feature_type=t5 --srcfilename=PSPire_test_ID.npz

python -m preprocess.extract_feat --feature_type=t5 --srcfilename=PSPire_test_noID.npz