import numpy as np,pandas as pd,pathlib,sys
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.base import clone
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run(path,output='outputs'):
 out=pathlib.Path(output);out.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(path);X=d.drop(columns=['serial_number','label','split','url_more_than_mean','domain_more_than_mean']);y=d.label;tr=d.split.eq('train');te=d.split.eq('test')
 pipe=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('select',SelectFromModel(LogisticRegression(penalty='l1',solver='liblinear',C=1,random_state=42),threshold=-np.inf)),('model',LogisticRegression(max_iter=3000,random_state=42))])
 g=GridSearchCV(pipe,[{'select__max_features':[5,10,15,20],'model__C':[.01,.1,1,10]},{'select':['passthrough'],'model__C':[.01,.1,1,10]}],scoring='f1',cv=StratifiedKFold(5,shuffle=True,random_state=42),refit=False,n_jobs=1);g.fit(X[tr],y[tr]);a=pd.DataFrame(g.cv_results_);a['k']=a.param_select__max_features.fillna(68).astype(int);a['C']=a.param_model__C.astype(float);a[['k','C','mean_test_score','std_test_score']].to_csv(out/'cv.csv',index=False)
 best=a.loc[a.k.lt(68)].sort_values(['mean_test_score','k'],ascending=[False,True]).iloc[0];m=clone(pipe).set_params(**best.params).fit(X[tr],y[tr]);names=X.columns[m['select'].get_support()];p=m.predict_proba(X[te])[:,1];pred=(p>=.5).astype(int);actual=y[te];tn,fp,fn,tp=confusion_matrix(actual,pred,labels=[0,1]).ravel();metrics=dict(k=len(names),C=float(best.C),cv_f1=float(best.mean_test_score),accuracy=accuracy_score(actual,pred),precision=precision_score(actual,pred),recall=recall_score(actual,pred),specificity=tn/(tn+fp),f1=f1_score(actual,pred),mcc=matthews_corrcoef(actual,pred),auc=roc_auc_score(actual,p),average_precision=average_precision_score(actual,p),tn=tn,fp=fp,fn=fn,tp=tp);print(metrics);pd.DataFrame([metrics]).to_csv(out/'metrics.csv',index=False)
 coef=pd.DataFrame({'feature':names,'coefficient':m['model'].coef_[0]}).sort_values('coefficient',key=abs,ascending=False);coef.to_csv(out/'features.csv',index=False);print(coef.to_string(index=False));d[['serial_number','split','label',*names]].to_csv(out/'selected_1000.csv',index=False);d.loc[te,['serial_number','label',*names]].assign(probability=p,prediction=pred).to_csv(out/'predictions.csv',index=False)
 for name,fnc in [('confusion',lambda:ConfusionMatrixDisplay.from_predictions(actual,pred)),('roc',lambda:RocCurveDisplay.from_predictions(actual,p)),('pr',lambda:PrecisionRecallDisplay.from_predictions(actual,p))]:
  fnc();plt.tight_layout();plt.savefig(out/(name+'.png'),dpi=150);plt.close()
if __name__=='__main__':run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'outputs')
