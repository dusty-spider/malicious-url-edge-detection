The dataset used in the project is made from combined open source licences datasets which author does not own, they can be downloaded:

----------------------------------------------------------------------------------------------

\-- Main dataset --

malicious - phishtank cisco phishing dataset, updated every 24hrs: https://phishtank.org

benign   - tranco domain dataset, updated every 24hrs: https://tranco-list.eu/ 

mixed    - fixed Kaggle dataset: https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset



\-- Data drift --

mixed    - phisuiil: https://www.kaggle.com/datasets/ndarvind/phiusiil-phishing-url-dataset


The main dataset was made by 
```
merging 3 datasets -> removing dublicates -> lowercase normalisation and preprocessing -> character level tokenisation -> vocab mapping, reproduction file creation -> class weighting -> 70/15/15 split application 
```