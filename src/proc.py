import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import datetime
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
con=psycopg2.connect(dbname= 'prod', host='lenda-data.cpzq9irxnxxa.us-west-2.redshift.amazonaws.com', port= '5439', user= 'wei', password= 'durinat3RALO')
df_l = pd.read_sql('select * from anondata.leads;', con=con)
df_o = pd.read_sql('select * from anondata.opportunities;', con=con)

feat_dt = [feat for feat, types in df_l.dtypes.iteritems() if str(types) == 'datetime64[ns]']
feat_o = [feat for feat, types in df_l.dtypes.iteritems() if str(types) == 'object']
feat_dt += [f for f in feat_o if 'datetime.datetime' in str(df_l[f].unique())]
obj = df_l[[f for f in feat_o if f not in feat_dt]].apply(pd.to_numeric, errors='ignore').dtypes
feat_f = list(obj[obj!='object'].index)
feat_o = list(obj[obj=='object'].index)
feat_o_manual = pd.read_csv('../data/lead_obj_am.csv')
feat_c = list(feat_o_manual[feat_o_manual.col_type=='d'].feature)
feat_n = list(feat_o_manual[feat_o_manual.col_type=='n'].feature)
feat_ff = list(feat_o_manual[feat_o_manual.col_type=='f'].feature)
feat_t = list(feat_o_manual[feat_o_manual.col_type=='t'].feature)
def create_label(X):
    label_id = df_o[(df_o.lock_won_date_c.notnull())&(df_o.locked_to_funded_c>0)&(df_o.stage_name=='Closed Won')].id
    return X['converted_opportunity_id'].apply(lambda x: 1 if x in list(label_id ) else 0)
def feattrans_c(df, cols):
    df_dum = pd.get_dummies(df[cols], prefix=cols)
    ndf = df.drop(cols, axis=1)
    return pd.concat([ndf, df_dum], axis = 1)
def feattrans_n(df, cols):
    ndf = df
    for col in cols:
        ndf[col] = ndf[col].notnull()
    return ndf
def text_process(text, col):
    vec = TfidfVectorizer(stop_words='english')
    vect = vec.fit_transform(text)
    pca = PCA()
    mat = pca.fit_transform(vect.todense())
    cols = [col+'_'+str(x) for x in xrange(mat.shape[1])]
    return pd.DataFrame(mat, columns=cols)
def feattrans_t(df, cols):
    ndf = df
    for col in cols:
        text = ndf[col]
        text_col = text_proc(text, col)
        ndf = ndf.drop(col, axis=1)
        ndf = pd.concat([ndf, text_col], axis = 1)
    return ndf
def feattrans_ddt(df, cols, col_benchmark):
    ndf = df
    cols = list(set(cols)-set([col_benchmark]))
    ndf['borrower_date_of_birth_c'] = ndf['borrower_date_of_birth_c'].replace(datetime.datetime(2973, 7, 16, 0, 0), datetime.datetime(1973, 7, 16, 0, 0))
    ndf[cols] = ndf[cols].astype('datetime64[ns]')
    for col in cols:
        ndf[col] = ndf[col].subtract(ndf[col_benchmark]).astype('timedelta64[h]')
    return ndf
def feattrans_dt(df, cols):
    ndf = df
    for col in cols:
        timefloat = pd.to_numeric(ndf[col], errors='ignore')
        timeline = pd.Series(MinMaxScaler().fit_transform(timefloat))
        year = ndf[col].apply(lambda x: x.year)
        month = ndf[col].apply(lambda x: x.month)
        day = ndf[col].apply(lambda x: x.day)
        dow = ndf[col].apply(lambda x: x.weekday())
        quarter = month.apply(lambda x: (x-1)//3+1)
        time = pd.concat([year, month, day, dow, quarter], axis=1)
        timecol=['year', 'month', 'day', 'dow', 'quarter']
        time.columns=[col+'_'+coln for coln in timecol]
        ndf = ndf.drop(col, axis=1)
        ndf = pd.concat([ndf, time], axis=1)
    return ndf
def feattrans_ff(df, cols):
    ndf = df
    def g(x):
        try:
            return float(str(x).translate(None, '$abcdefghijklmnopqrstuvwxyz, '))
        except:
            return x
    ndf[cols] = ndf[cols].applymap(g)
    return ndf
def feattrans_f(df, cols):
    df[cols] = df[cols].apply(pd.to_numeric, errors='ignore')
    return df
def remove_feat(df, cols):
    obj=[]
    for col, typ in df.dtypes.iteritems():
        if str(typ) == 'object':
            obj.append(col)
    cols += obj
    return df.drop(cols, axis=1)
def missing_data(df):
    df = df.fillna(df.mean())
    df = df.fillna(0)
    '''try do better!'''
    return df
def full_columns(df):
    DF = df.notnull().sum()==len(df)
    full = [name for name in DF.index if DF[name] == True]
    return df[full]
def Pipline_DataCleaning(datainput):
    cols = list(datainput.columns)
    pdt = [feat for feat in cols if feat in feat_dt]
    pf = [feat for feat in cols if feat in feat_f]
    pff = [feat for feat in cols if feat in feat_ff]
    pc = [feat for feat in cols if feat in feat_c]
    pn = [feat for feat in cols if feat in feat_n]
    if 'created_date' in cols:
        if len(pdt)>1:
            datainput = feattrans_ddt(datainput, pdt, 'created_date')
        datainput = feattrans_dt(datainput, ['created_date'])
    if len(pf)>0:
        datainput = feattrans_f(datainput, pf)
    if len(pff)>0:
        datainput = feattrans_ff(datainput, pff)
    if len(pc)>0:
        datainput = feattrans_c(datainput, pc)
    if len(pn)>0:
        datainput = feattrans_n(datainput, pn)
    datainput = remove_feat(datainput, [])
    datainput = missing_data(datainput)
    #datainput = full_columns(datainput)
    return datainput

def Pipline_Model(X, y, classifier = 'RF'):
    model_rf = RandomForestClassifier(class_weight='balanced')
    result_rf = model_rf.fit(X, y)
    feat = result_rf.feature_importances_
    model_lr = LogisticRegression(class_weight='balanced')
    result_lr = model_lr.fit(X, y)
    coeff = result_lr.coef_.reshape(-1)
    feat_imp = sorted(zip(X.columns, feat, coeff), key=lambda x: x[1])[::-1]
    if classifier == "RF":
        score_model = model_rf
    elif classifier == 'LR':
        score_model = model_lr
    elif classifier == 'GNB':
        score_model = GaussianNB()
    elif classifier == 'SVM':
        score_model = SVC()
    accuracy = cross_val_score(model_rf, X, y, scoring = 'accuracy', cv = 5)
    f1 = cross_val_score(model_rf, X, y, scoring = 'f1', cv = 5)
    precision = cross_val_score(model_rf, X, y, scoring = 'precision', cv = 5)
    recall = cross_val_score(model_rf, X, y, scoring = 'recall', cv = 5)
    roc_auc = cross_val_score(model_rf, X, y, scoring = 'roc_auc', cv = 5)
    return result_rf, feat_imp, accuracy, f1, precision, recall, roc_auc
def get_stage_feat(df_l, stages):
    # stages = ['results','about_you','about_home','employment_and_income','assets','government_monitoring_questions','declarations','credit_check_permission','credit_check','sign','upload','complete']
    #mask = (df_l.furthest_step_c.isnull())&(create_label(df_l)==1)
    #df_l['furthest_step_c'][mask] = stages[-1]
    conditions=[]
    for stage in stages:
        conditions.append(df_l.furthest_step_c==stage)
    groups=[]
    for i in xrange(len(conditions)):
        mask = pd.Series([False]*len(df_l))
        for j in xrange(i,len(conditions)):
            mask += conditions[j]
        groups.append(df_l[mask])
    count_ = Counter(df_l.furthest_step_c)
    group_feat ={}
    for stage in stages:
        x = df_l[df_l.furthest_step_c==stage].notnull().sum()
        ind = x[x.where(x>=.90*count_[stage]).notnull()].index
        group_feat[stage] = set(ind)
    stage_feat = {}
    for i in xrange(1, len(stages)):
        stage_feat[stages[i]] = (list(group_feat[stages[i]] - group_feat[stages[i-1]]))
    stage_feat['results'] = [
    'converted_opportunity_id', 'email','id', 'received_at', 'uuid', 'uuid_ts',
    'created_by_id', 'created_date', 'created_hour_c', 'days_since_created_c',
    'full_lead_c', 'gross_revenue_c', 'is_unread_by_owner', 'landing_page_was_hubspot_c',
    'last_modified_by_id', 'last_modified_date', 'lead_number_c', 'lead_score_c',
    'lead_score_completeness_c', 'name', 'owner_id', 'photo_url', 'purchase_c',
    'sales_lead_as_number_c', 'second_mortgage_c','status','subordinate_second_mortgage_c',
    'system_modstamp', 'voicemail_unavailable_c', 'working_status_c'
    ]
    stage_feat['employment_and_income'] += []
    stage_feat['assets'] += ['borrower_current_employment_length_month_c', 'borrower_employer_contact_name_c']
    stage_feat['government_monitoring_questions'] = list(set(stage_feat['government_monitoring_questions']) - set(['number_of_pageviews_c', 'borrower_current_employment_length_month_c', 'selected_rate_p_i_payment_c', 'product_type_c']))
    stage_feat['declarations'] = ['borrower_ethnicity_c', 'borrower_gender_c']
    stage_feat['credit_check_permission'] = []
    stage_feat['sign'] = []
    return stage_feat
def get_groups(df_l, stages):
    # stages = ['results','about_you','about_home','employment_and_income','assets','government_monitoring_questions','declarations','credit_check_permission','credit_check','sign','upload','complete']
    #mask = (df_l.furthest_step_c.isnull())&(create_label(df_l)==1)
    #df_l['furthest_step_c'][mask] = stages[-1]
    conditions=[]
    for stage in stages:
        conditions.append(df_l.furthest_step_c==stage)
    groups={}
    for i in xrange(len(conditions)):
        mask = pd.Series([False]*len(df_l))
        for j in xrange(i,len(conditions)):
            mask += conditions[j]
        groups[stages[i]] = df_l[mask]
    return groups
def get_stage_feat_agg(df, stages):
    stage_feat = get_stage_feat(df, stages)
    stage_feat_agg = {}
    for i in xrange(len(stages)):
        feats = []
        for j in xrange(i+1):
            feats += stage_feat[stages[j]]
        stage_feat_agg[stages[i]] = feats
    return stage_feat_agg
def classify_step(dft, stages, stage_feat_agg):
    dft['converted_opportunity_id'] = dft['converted_opportunity_id'].fillna(0)
    for test_stage in stage_feat_agg:
        df_test = dft[stage_feat_agg[test_stage]]
        dft[test_stage] = df_test.apply(lambda x: x.notnull()).mean(axis=1)
    def f(series):
        if series['furthest_step_c'] in stages:
            return series['furthest_step_c']
        else:
            for stage in stages[::-1]:
                if series[stage]>0.8:
                    return stage
            return 'results'
    dft.furthest_step_c = dft.apply(lambda x: f(x), axis=1)
    dft = dft.drop(stages, axis=1)
    return dft
