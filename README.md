# BiXGBoost: a scalable, flexible boosting based method for reconstructing gene regulatory networks
BiXGBoost is a bidirectional method that considers time information and integrates multi decision trees by the boosting method.
## Dependency
Xgboost Version=0.6 [Reference Link](https://xgboost.readthedocs.io/en/latest/build.html "悬停显示")
### Pip install
    Python version=3.5
    scikit-learn Version >= 0.18
    Pandas >= 0.19.x
    Numpy >=1.12.x

The install of python and packages (except Xgboost) can be quickly done by [Anaconda 5](https://www.anaconda.com/download/ "悬停显示")

## Example
from BiXGBoost import *

mainRun(expressionFile,samples,outputfile,p_lambda=0,p_alpha=1,maxlag=2,timelag=2)
### parameters
    expressionFile: path of the expression file
    samples: the number of samples in the expresssionFile on same time piont, such like 10 in DREAM4 InSilico_Size100
    outputfile: path of output file
    p_lambda & p_alpha: the L1 and L2 regularization parameters in XGBoost
    timelag: the maximum lag of time point
       

