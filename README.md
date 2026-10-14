import os; os.makedirs("figures", exist_ok=True)
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import *
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
R={}
df=pd.read_csv('loan_approval_dataset.csv')
df.columns=df.columns.str.strip()
for c in df.select_dtypes('object').columns.tolist()+[c for c in df.columns if df[c].dtype.name=='str']:
    df[c]=df[c].str.strip()
R['raw_shape']=list(df.shape); R['missing']=int(df.isnull().sum().sum()); R['dups']=int(df.drop(columns='loan_id').duplicated().sum())
df=df.drop(columns='loan_id')
R['clean_shape']=list(df.shape)
vc=df.loan_status.value_counts(); R['approved']=int(vc['Approved']); R['rejected']=int(vc['Rejected'])
R['desc']=df.describe().T.round(2).to_dict('index')
R['edu']=df.education.value_counts().to_dict(); R['selfemp']=df.self_employed.value_counts().to_dict()
# outliers (IQR) on income/loan amount
for c in ['income_annum','loan_amount','cibil_score','luxury_assets_value']:
    q1,q3=df[c].quantile([.25,.75]); i=q3-q1
    R['out_'+c]=int(((df[c]<q1-1.5*i)|(df[c]>q3+1.5*i)).sum())
# negative asset check
R['neg_assets']=int((df[['residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']]<0).sum().sum())
# feature engineering
df['total_assets']=df[['residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']].sum(axis=1)
df['loan_to_income']=df.loan_amount/df.income_annum
df['loan_to_assets']=df.loan_amount/df.total_assets.replace(0,np.nan)
df['payment_to_income']=df.loan_amount/(df.loan_term*df.income_annum)
df['loan_to_assets']=df['loan_to_assets'].fillna(df['loan_to_assets'].median())
df['education']=(df.education=='Graduate').astype(int)  # 1 Graduate
df['self_employed']=(df.self_employed=='Yes').astype(int)
df['target']=(df.loan_status=='Rejected').astype(int)   # 1 = rejected / high risk
df=df.drop(columns='loan_status')
R['eng_desc']=df[['total_assets','loan_to_income','loan_to_assets','payment_to_income']].describe().T.round(3).to_dict('index')
corr=df.corr(); R['corr_target']=corr['target'].drop('target').round(3).sort_values().to_dict()
R['corr_pairs']={'loan_amount~income':round(corr.loc['loan_amount','income_annum'],3),'loan_amount~luxury':round(corr.loc['loan_amount','luxury_assets_value'],3),'income~luxury':round(corr.loc['income_annum','luxury_assets_value'],3),'income~total_assets':round(corr.loc['income_annum','total_assets'],3),'loan_to_income~payment_to_income':round(corr.loc['loan_to_income','payment_to_income'],3)}
R['mean_by_target']=df.groupby('target')[['cibil_score','income_annum','loan_amount','loan_to_income','payment_to_income','loan_to_assets']].mean().round(3).to_dict('index')
R['rej_rate_by_edu']=df.groupby('education').target.mean().round(3).to_dict()
R['rej_rate_by_se']=df.groupby('self_employed').target.mean().round(3).to_dict()
R['cibil_bins']=df.groupby(pd.cut(df.cibil_score,[299,500,600,700,900])).target.mean().round(3).to_dict()
R['cibil_bins']={str(k):v for k,v in R['cibil_bins'].items()}
# ---- figures
sns.set_theme(style='whitegrid'); lab={0:'Approved',1:'Rejected'}
plt.figure(figsize=(6,4)); ax=sns.countplot(x=df.target.map(lab),order=['Approved','Rejected'],palette=['#2b7bba','#e07b39']); 
for p in ax.patches: ax.annotate(int(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),ha='center',va='bottom')
plt.title('Loan Decision Distribution');plt.xlabel('Loan Status');plt.ylabel('Number of Applicants');plt.tight_layout();plt.savefig('figures/f1.png',dpi=200);plt.close()
plt.figure(figsize=(6,4));sns.histplot(df.income_annum/1e5,bins=25,kde=True,color='#2b7bba');plt.title('Annual Income Distribution');plt.xlabel('Annual Income (in lakhs)');plt.ylabel('Number of Applicants');plt.tight_layout();plt.savefig('figures/f2.png',dpi=200);plt.close()
plt.figure(figsize=(6,4));sns.boxplot(x=df.target.map(lab),y=df.cibil_score,order=['Approved','Rejected'],palette=['#2b7bba','#e07b39']);plt.title('CIBIL Score vs Loan Decision');plt.xlabel('Loan Status');plt.ylabel('CIBIL Score');plt.tight_layout();plt.savefig('figures/f3.png',dpi=200);plt.close()
plt.figure(figsize=(6,4));sns.boxplot(x=df.target.map(lab),y=df.loan_to_income,order=['Approved','Rejected'],palette=['#2b7bba','#e07b39']);plt.title('Loan-to-Income Ratio vs Loan Decision');plt.xlabel('Loan Status');plt.ylabel('Loan-to-Income Ratio');plt.tight_layout();plt.savefig('figures/f4.png',dpi=200);plt.close()
cm_cols=['no_of_dependents','education','self_employed','income_annum','loan_amount','loan_term','cibil_score','residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value','loan_to_income','loan_to_assets','payment_to_income','target']
plt.figure(figsize=(9,7.5));sns.heatmap(df[cm_cols].corr(),annot=True,fmt='.2f',cmap='coolwarm',annot_kws={'size':6},cbar_kws={'shrink':.8});plt.title('Correlation Heatmap');plt.xticks(rotation=60,ha='right',fontsize=7);plt.yticks(fontsize=7);plt.tight_layout();plt.savefig('figures/f5.png',dpi=200);plt.close()
# ---- modelling
X=df.drop(columns=['target','total_assets']); y=df.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
R['split']={'train':len(Xtr),'test':len(Xte),'train_pos':int(ytr.sum()),'test_pos':int(yte.sum()),'test_neg':int((1-yte).sum()),'features':list(X.columns)}
spw=(ytr==0).sum()/(ytr==1).sum()
rf=RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1)
xg=XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=5,subsample=.8,colsample_bytree=.8,scale_pos_weight=spw,eval_metric='logloss',random_state=42,n_jobs=-1)
lg=LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,class_weight='balanced',random_state=42,n_jobs=-1,verbose=-1)
ens=VotingClassifier([('rf',rf),('xgb',xg),('lgbm',lg)],voting='soft',n_jobs=1)
models={'Random Forest':rf,'XGBoost':xg,'LightGBM':lg,'Soft-Voting Ensemble':ens}
res={};probs={};cms={}
for n,m in models.items():
    m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]; pr=(p>=.5).astype(int); probs[n]=p
    res[n]=dict(acc=accuracy_score(yte,pr),prec=precision_score(yte,pr),rec=recall_score(yte,pr),f2=fbeta_score(yte,pr,beta=2),auc=roc_auc_score(yte,p),f1=f1_score(yte,pr))
    cms[n]=confusion_matrix(yte,pr)
R['res']={k:{a:round(b,4) for a,b in v.items()} for k,v in res.items()}
R['cm']={k:v.tolist() for k,v in cms.items()}
R['spw']=round(spw,3)
# CV on training data
skf=StratifiedKFold(5,shuffle=True,random_state=42)
cvr={}
for n,m in models.items():
    a=cross_val_score(m,Xtr,ytr,cv=skf,scoring='roc_auc',n_jobs=1); 
    f=cross_val_score(m,Xtr,ytr,cv=skf,scoring=make_scorer(fbeta_score,beta=2),n_jobs=1)
    cvr[n]={'auc_mean':round(a.mean(),4),'auc_std':round(a.std(),4),'f2_mean':round(f.mean(),4),'f2_std':round(f.std(),4)}
R['cv']=cvr
# fig6 confusion matrices
fig,axs=plt.subplots(2,2,figsize=(8,6.5))
for ax,(n,c) in zip(axs.ravel(),cms.items()):
    sns.heatmap(c,annot=True,fmt='d',cmap='Blues',ax=ax,xticklabels=['Approved','Rejected'],yticklabels=['Approved','Rejected'],cbar=False);ax.set_title(n+' Confusion Matrix');ax.set_xlabel('Predicted');ax.set_ylabel('Actual')
plt.tight_layout();plt.savefig('figures/f6.png',dpi=200);plt.close()
plt.figure(figsize=(6.5,5))
for n,p in probs.items():
    f,t,_=roc_curve(yte,p);plt.plot(f,t,label=f'{n} (AUC = {res[n]["auc"]:.3f})',lw=2 if 'Ens' in n else 1.4)
plt.plot([0,1],[0,1],'k--',lw=1);plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate');plt.title('ROC Curves of the Models');plt.legend(loc='lower right');plt.tight_layout();plt.savefig('figures/f7.png',dpi=200);plt.close()
# comparison bar (extra)
# SHAP: use LightGBM (fast, tree) and also XGB
best=max(['Random Forest','XGBoost','LightGBM'],key=lambda k:res[k]['auc']); R['shap_model']=best
sm=models[best]
ex=shap.TreeExplainer(sm); sv=ex(Xte)
vals=sv.values
if vals.ndim==3: sv=shap.Explanation(values=vals[:,:,1],base_values=sv.base_values[:,1],data=sv.data,feature_names=list(Xte.columns))
plt.figure();shap.summary_plot(sv.values,Xte,show=False,plot_size=(8,5.5));plt.title(f'SHAP Summary Plot ({best})');plt.tight_layout();plt.savefig('figures/f8.png',dpi=200,bbox_inches='tight');plt.close()
imp=pd.Series(np.abs(sv.values).mean(0),index=Xte.columns).sort_values(ascending=False); R['shap_rank']=imp.round(4).head(8).to_dict()
# applicant-level: pick a rejected applicant correctly flagged with high prob
idx=int(np.argmax(np.where(yte.values==1,probs[best],0)))
plt.figure();shap.plots.waterfall(sv[idx],max_display=8,show=False);plt.title('SHAP explanation for one rejected applicant');plt.tight_layout();plt.savefig('figures/f9.png',dpi=200,bbox_inches='tight');plt.close()
R['applicant']={k:(float(v) if isinstance(v,(int,float,np.number)) else v) for k,v in Xte.iloc[idx].items()}; R['applicant_prob']=float(probs[best][idx])
top=pd.Series(sv[idx].values,index=Xte.columns).sort_values(key=abs,ascending=False).head(4); R['applicant_top']=top.round(3).to_dict()
# base-vs-ens thresholds: recall @ various? skip
json.dump(R,open('results.json','w'),indent=1,default=str)
print(json.dumps({k:R[k] for k in ['res','cm','cv','shap_model','shap_rank','split','corr_target','spw']},indent=1))
