### Install kaggle API

```
pip install kaggle
```
document: https://github.com/Kaggle/kaggle-api

### Data download

```
cd ./m5forecasting/data/
kaggle competitions download -c m5-forecasting-accuracy
unzip m5-forecasting-accuracy.zip
```

### Model training
```
cd ./m5forecasting/src/
python entry.py lgbm lgbm1914
python entry.py lgbm lgbm1900
python entry.py lgbm lgbm1886
python entry.py lgbm lgbm1872
python entry.py lgbm lgbm1858
python entry.py lgbm lgbm1942
```

### Model ensemble & make submission
python entry.py ensemble lgbm1942