import sys, pathlib, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import matplotlib.pyplot as plt

def run(path, out='outputs_20'):
 out=pathlib.Path(out); out.mkdir(exist_ok=True,parents=True)
 df=pd.read_csv(path).drop(columns=['split'],errors='ignore') if str(path).endswith('.csv') else pd.concat([g.sample(500,random_state=42) for _,g in pd.read_excel(path).drop_duplicates('url').groupby('label')]).sample(frac=1,random_state=42).reset_index(drop=True)
 X=df.drop(columns=['serial_number','url','label','url_more_than_mean','domain_more_than_mean'],errors='ignore'); y=df.label
 tr,te=train_test_split(np.arange(len(df)),test_size=.2,stratify=y,random_state=42)
 pipe=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('select','passthrough'),('model',LogisticRegression(max_iter=3000,random_state=42))])
 selector=SelectFromModel(LogisticRegression(penalty='l1',solver='liblinear',C=1,random_state=42),threshold=-np.inf)
 grid=GridSearchCV(pipe,[{'select':[selector],'select__max_features':[5,10,15,20],'model__C':[.01,.1,1,10]}],scoring='f1',cv=StratifiedKFold(5,shuffle=True,random_state=42),n_jobs=1)
 grid.fit(X.iloc[tr],y.iloc[tr]); m=grid.best_estimator_; p=m.predict_proba(X.iloc[te])[:,1]; pred=(p>=.5).astype(int); actual=y.iloc[te]; tn,fp,fn,tp=confusion_matrix(actual,pred,labels=[0,1]).ravel()
 names=X.columns if m['select']=='passthrough' else X.columns[m['select'].get_support()]; coef=pd.DataFrame({'feature':names,'coefficient':m['model'].coef_[0]}); coef=coef.reindex(coef.coefficient.abs().sort_values(ascending=False).index); coef.to_csv(out/'coefficients.csv',index=False)
 results=dict(accuracy=accuracy_score(actual,pred),precision=precision_score(actual,pred),recall=recall_score(actual,pred),specificity=tn/(tn+fp),f1=f1_score(actual,pred),mcc=matthews_corrcoef(actual,pred),auc=roc_auc_score(actual,p),average_precision=average_precision_score(actual,p),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),cv_f1=grid.best_score_,k=len(names),C=grid.best_params_['model__C']); print(json.dumps(results,indent=2)); (out/'metrics.json').write_text(json.dumps(results,indent=2)); pd.DataFrame([results]).to_csv(out/'metrics.csv',index=False)
 df.drop(columns='url',errors='ignore').assign(split=np.where(np.isin(np.arange(len(df)),te),'test','train')).to_csv(out/'sample_1000.csv',index=False); df[['serial_number','label']].iloc[te].assign(probability=p,prediction=pred).to_csv(out/'predictions.csv',index=False)
 cv=pd.DataFrame(grid.cv_results_); cv['k']=cv['param_select__max_features'].fillna(X.shape[1]); cv[['k','param_model__C','mean_test_score','std_test_score']].to_csv(out/'cv.csv',index=False)
 for name,plot in [('confusion',lambda:ConfusionMatrixDisplay.from_predictions(actual,pred)),('roc',lambda:RocCurveDisplay.from_predictions(actual,p)),('pr',lambda:PrecisionRecallDisplay.from_predictions(actual,p))]:
  plot(); plt.tight_layout(); plt.savefig(out/(name+'.png'),dpi=150); plt.close()
 ax=coef.head(10).iloc[::-1].plot.barh(x='feature',y='coefficient',legend=False,color='#138a8a',figsize=(8,4.5)); plt.tight_layout(); plt.savefig(out/'coefficients.png',dpi=150); plt.close()
 return results
if __name__=='__main__': run(sys.argv[1])
