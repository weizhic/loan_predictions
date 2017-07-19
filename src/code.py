import numpy as np
import pandas as pd
import psycopg2
import proc
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    con=psycopg2.connect(dbname= 'prod', host='lenda-data.cpzq9irxnxxa.us-west-2.redshift.amazonaws.com', port= '5439', user= 'wei', password= 'durinat3RALO')
    df = pd.read_sql('select * from anondata.leads;', con=con)

    stages = ['results','about_you','about_home','employment_and_income','assets','government_monitoring_questions','declarations','credit_check_permission','credit_check','sign','upload','complete']
    stage_feat = proc.get_stage_feat(df, stages)
    stage_feat_agg = proc.get_stage_feat_agg(stage_feat, stages)
    # Finding top features for each stage
    groups = proc.get_groups(df, stages)
    feat_imp={}
    for stage in stages:
        X = groups[stage][['converted_opportunity_id']+stage_feat[stage]]
        y = proc.create_label(X)
        X_clean = proc.Pipline_DataCleaning(X)

        ros = RandomOverSampler()

        X_res, y_res = ros.fit_sample(X_clean, y)
        X_res_pd = pd.DataFrame(X_res, columns = X_clean.columns)

        rf = RandomForestClassifier()

        model = rf.fit(X_res_pd, y_res)
        feat = model.feature_importances_

        feat_imp[stage] = sorted(zip(X_res_pd.columns, feat), key=lambda x: x[1])[::-1]



    # Build model for conversion prediction
    X = df
    y = proc.create_label(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    groups = proc.get_groups(X_train, stages)
    models = {}
    cross_val = {}
    for stage in stages:
        X = groups[stage][stage_feats]
        y = proc.create_label(X)
        X_clean = proc.Pipline_DataCleaning(X)

        ros = RandomOverSampler()

        X_res, y_res = ros.fit_sample(X_clean, y)
        X_res_pd = pd.DataFrame(X_res, columns = X_clean.columns)

        rf = RandomForestClassifier()

        model = rf.fit(X_res_pd, y_res)

        models[stage] = model
        cross_val[stage] = cross_val_score(model, X_res_pd, y_res)

    # Testing
