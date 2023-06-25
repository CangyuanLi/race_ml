cp src/data/final/flz_train.parquet upload/src/data/final/
cp src/data/final/fl_train.parquet upload/src/data/final/

cp -R src/utils upload/src/

cp src/train_fl_bi.py upload/src/
cp src/train_flz_bi.py upload/src/
cp src/keras_models.py upload/src/

for f in src/train*
    cp $f upload/src/
end

cp src/models.py upload/src/
cp src/data.py upload/src
