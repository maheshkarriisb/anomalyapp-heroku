import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import math
import pickle

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
from collections import Counter

from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import os
basePath = os.getcwd()

# aggregateOptions = ['Record Level',"Day Level"]['Record Level',"Day Level"]
aggregateOptions = ['Record Level']

class ModelPipeline:
    def __init__(self, modelName, aggregation_level, net_amount_quantiles, label_encoders, column_transformers, model):
        self.modelName = modelName
        self.aggregation_level=aggregation_level
        self.net_amount_quantiles = net_amount_quantiles
        self.label_encoders = label_encoders
        self.column_transformers = column_transformers
        self.model = model

    def saveModel(self, aggregation_level):
        with open("easyInput/modelPipelines/{}/{}".format(aggregation_level,self.modelName), 'wb') as f:
            pickle.dump(self, f)
            f.close()


def get_outliers(df):
    featureColumns = ['company_ref_id',
              'cost_center_id',
              'cost_center_name',
              'crc_name',
              'ledger_acct_name',
              'budget_item_id',
              'wo_proj_id',
              'facility_id',
              'journal_name',
              'journal_source']
    otherReqColumns = ['net_amount']
    mean_amt = df.net_amount.mean()
    std_amt = df.net_amount.std()

    left_hinge1 = (mean_amt-(2*std_amt))
    right_hinge1 = (mean_amt+(2*std_amt))

    median_amt = df.net_amount.quantile(0.5)
    iqr_amt = df.net_amount.quantile(0.95) - df.net_amount.quantile(0.05)
    left_hinge2 = (median_amt-(1.5*iqr_amt))
    right_hinge2 = (median_amt+(1.5*iqr_amt))

    rolling_mean = df.groupby(featureColumns)['net_amount'].transform((lambda s: s.rolling(6).mean()))
    rolling_std = df.groupby(featureColumns)['net_amount'].transform((lambda s: s.rolling(6).std()))

    expanding_mean = df.groupby(featureColumns)['net_amount'].transform((lambda s: s.expanding().mean()))
    expanding_std = df.groupby(featureColumns)['net_amount'].transform((lambda s: s.expanding().std()))

    left_hinges1 = (rolling_mean - 3*rolling_std).fillna(-1e12)
    right_hinges1 = (rolling_mean + 3*rolling_std).fillna(1e12)

    left_hinges3 = (expanding_mean - 3*expanding_std ).fillna(-1e12)
    right_hinges3 = (expanding_mean + 3*expanding_std ).fillna(1e12)

    amt_mean = df.groupby(featureColumns)['net_amount'].transform('mean')
    amt_std = df.groupby(featureColumns)['net_amount'].transform('std')
    left_hinges2 = (amt_mean - 3*amt_std).fillna(-1e12)
    right_hinges2 = (amt_mean + 3*amt_std).fillna(1e12)

    global_outlier_serious = df.net_amount.apply(lambda x: 1 if (x<left_hinge1) or (x>right_hinge1) else 0)
    global_outlier_mild = df.net_amount.apply(lambda x: 1 if (x<left_hinge2) or (x>right_hinge2) else 0)
    local_outlier_serious = ((df.net_amount<left_hinges2) | (df.net_amount>right_hinges2))*1
    local_outlier_mild = ((df.net_amount<left_hinges1) | (df.net_amount>right_hinges1) | (df.net_amount<left_hinges3) | (df.net_amount>right_hinges3))*1

    # with open("easyInput/isolationForestModel.pkl", "rb") as f:
    #     i_forest = pickle.load(f)
    #     f.close()

    from sklearn.ensemble import IsolationForest

    par_n_estimators = 10
    par_max_samples = "auto"
    par_contamination = 0.01
    par_bootstrap = False
    par_n_jobs = -1
    par_max_features = 2
    i_forest = IsolationForest()
    i_forest.fit(df[featureColumns+otherReqColumns])

    anomaly_score = i_forest.decision_function(df.loc[:,featureColumns+otherReqColumns])
    anomaly_threshold = None
    # for thresh,cnt in Counter(anomaly_score).most_common():
    #     if thresh>0.1:
    #         anomaly_threshold = thresh
    counter_thresh = pd.DataFrame(list(Counter(anomaly_score).most_common()), columns=['thresh','cnt']).sort_values('thresh',ascending=False).reset_index(drop=True)
    counter_thresh['cumCnt'] = counter_thresh['cnt']
    print (counter_thresh[:20])
    counter_thresh = counter_thresh[counter_thresh['cumCnt']>=0.05*len(counter_thresh)]
    anomaly_threshold = counter_thresh['thresh'].values[0]
    outliers_isoForest = (anomaly_score>=anomaly_threshold)*1
    print ("Num Rows :", len(df))
    print ("Num global_outlier_serious :", np.sum(global_outlier_serious))
    print ("Num global_outlier_mild :", np.sum(global_outlier_mild))
    print ("Num local_outlier_serious :", np.sum(local_outlier_serious))
    print ("Num local_outlier_mild :", np.sum(local_outlier_mild))
    print ("Num outliers_isoForest :", np.sum(outliers_isoForest))

    return global_outlier_serious,global_outlier_mild,local_outlier_serious,local_outlier_mild,outliers_isoForest



def input_dataset():
    load_options, datasets = dict(), dict()
    load_options["pre_dataset"] = st.checkbox(
        "Load an existing dataset", True
    )
    dataset_config = {'April 2021 data':'{}/easyInput/Apr21Data.csv'.format(basePath)}
    if load_options["pre_dataset"]:
        dataset_name = st.selectbox(
            "Select an already existing dataset",
            options=['April 2021 data'],
            )
        df = pd.read_csv(dataset_config[dataset_name])
        load_options["dataset"] = dataset_name
        load_options["separator"] = ","
    else:
        file = st.file_uploader(
            "Upload a csv file", type="csv"
        )
        load_options["separator"] = st.selectbox(
            "What is the separator?", [",", ";", "|"]
        )

        if file:
            df = pd.read_csv(file, sep=load_options["separator"])
        else:
            st.stop()
    datasets["uploaded"] = df.copy()
    return df, load_options, datasets

def app():

    featureColumns = ['company_ref_id',
              'cost_center_id',
              'cost_center_name',
              'crc_name',
              'ledger_acct_name',
              'budget_item_id',
              'wo_proj_id',
              'facility_id',
              'journal_name',
              'journal_source']
    otherReqColumns = ['net_amount']

    st.header("Model Playground")
    st.write("Build your own models and deploy!")
    with st.expander("What does the model playground do?", expanded=False):
        st.write("The playground allows you to build your own model!")
        st.write("* Select/Upload the data")
        st.write("* Select the options in modeling/sampling")
        st.write("* Run model and evaluate performance")
    st.write("")

    st.sidebar.title("1. Select/Upload Data")

    # Load data
    with st.sidebar.expander("Dataset", expanded=True):
        df, load_options, datasets = input_dataset()
        df = df.sort_values('key_asof').reset_index(drop=True)

    aggregation_level = st.sidebar.selectbox("Select the aggregation level", options = aggregateOptions)
    if aggregation_level == 'Day Level':
        allData = df.groupby(['key_asof']+featureColumns).sum().reset_index()#.dropna()
        net_amount_quantiles = pd.read_pickle("easyInput/net_amount_quantiles_on_agg_data.pkl")
    else:
        allData = df.loc[:, ['key_asof']+featureColumns+otherReqColumns]#.dropna()
        net_amount_quantiles = pd.read_pickle("easyInput/net_amount_quantiles.pkl")
    st.write(allData)

    # with open("easyInput/labelEncoders.pkl",'rb') as f:
    #     le_list = pickle.load(f)
    #     f.close()
    le_list = dict()
    for col in featureColumns:
        try:
            print(col)
            le_model = LabelEncoder()
            allData[col] = le_model.fit_transform(allData[col])
            le_list[col] = le_model
        except Exception as E:
            print ("*"*100)
            print("Error in Label Encoder : ", col)
            print(allData.loc[pd.isna(allData[col]),col])
            print(pd.isna(allData[col]).sum(),len(allData[col]))
            raise(E)


    allOutliers = dict()
    allOutliers['global_outlier_serious'], allOutliers['global_outlier_mild'], allOutliers['local_outlier_serious'], allOutliers['local_outlier_mild'], allOutliers['outliers_isoForest'] = get_outliers(allData)
    outlierOptions = dict()
    st.sidebar.write("Select the statistical outlier to include for validation/model")
    for k in allOutliers.keys():
        outlierOptions[k]=st.sidebar.checkbox(k, True)
    print (outlierOptions)
    outlierDf = pd.DataFrame(dict([(k,allOutliers[k]) for k in allOutliers.keys() if outlierOptions[k]]))
    label = outlierDf.max(axis=1)
    print("Total Num Outliers : ", label.sum())

    st.sidebar.title("2. Modelling")

    model_type = st.sidebar.selectbox("Select the model you want to build", options=['Semi-Supervised','Autoencoder'])
    split_ratio = st.sidebar.slider(
            "Set the train test split",
            min_value=0.3,
            max_value=0.95,
            step=0.005,
            value=0.8,
        )

    if model_type=='Autoencoder':
        allData['net_amount_q'] = pd.cut(allData['net_amount'], bins=list(net_amount_quantiles), labels = list(range(len(net_amount_quantiles)-1)))
        ohe = OneHotEncoder(sparse=False)
        allDataAutoEncoder = ohe.fit_transform(allData.loc[:,featureColumns+['net_amount_q']])
        train_data = allDataAutoEncoder[:int(len(allDataAutoEncoder)*split_ratio)]
        test_data = allDataAutoEncoder[int(len(allDataAutoEncoder)*split_ratio):]
    elif model_type=='Semi-Supervised':
        modelId = st.sidebar.selectbox("Select Model to use: ", options=['LightGBM', 'RandomForest', 'CatBoost', 'AdaBoost','GBTree'])
        k_fold = st.sidebar.selectbox("Select Number of folds CV: ", options=[2,3,4,5,6],index=2)

        allDataSupervised = allData.loc[:,featureColumns+['net_amount']].copy()
        for col in featureColumns:
            allDataSupervised[col] = allDataSupervised[col].astype('category')
        if modelId in ['RandomForest', 'CatBoost', 'AdaBoost','GBTree']:
            ct_model = ColumnTransformer( [('one_hot_encoder', OneHotEncoder(sparse=False), featureColumns)], remainder='passthrough')
            allDataSupervised = ct_model.fit_transform(allDataSupervised)
        else:
            ct_model = None
        train_data = allDataSupervised[:int(len(allDataSupervised)*split_ratio)]
        train_label = label[:int(len(allDataSupervised)*split_ratio)]

        test_data = allDataSupervised[int(len(allDataSupervised)*split_ratio):]
        test_label = label[int(len(allDataSupervised)*split_ratio):]

        # kf = KFold(int(k_fold))
        # for train_idx, val_idx in kf.split(train_data, train_label):
        #     xTrain = train_data[train_idx]
        #     yTrain = train_label[train_idx]
        #     xVal = train_data[val_idx]
        #     yVal = train_label[val_idx]
        #     lgb_model = LGBMClassifier(n_estimators=3000)

        if modelId=='LightGBM':
            model_obj = LGBMClassifier(n_jobs=-1)
        elif modelId=='RandomForest':
            model_obj = RandomForestClassifier(n_estimators=10)
        elif modelId =='CatBoost':
            model_obj=CatBoostClassifier(iterations=200)
        elif modelId == 'AdaBoost':
            model_obj = AdaBoostClassifier(n_estimators=20)
        elif modelId=='GBTree':
            model_obj = GradientBoostingClassifier(n_estimators=25)

        scores = cross_val_score(model_obj, train_data, train_label, scoring='roc_auc', cv=int(k_fold), n_jobs=-1)
        st.write("Cross Validation ROC_AUC - Avg & std : ",np.mean(scores), np.std(scores))

        model_obj.fit(train_data, train_label)
        test_preds = model_obj.predict(test_data)
        test_probs = model_obj.predict_proba(test_data)[:,1]

        predictionDf = pd.DataFrame({'actual':test_label, 'predicted':test_preds})
        st.write(predictionDf.groupby(["actual","predicted"]).size().reset_index())

        st.write("* Accuracy : {}\n\n* F1-Score: {}\n\n* AUC-ROC Score: {}".format(accuracy_score(test_label,test_preds), f1_score(test_label, test_preds), roc_auc_score(test_label, test_probs)))

        # st.write("### Save to production")
        if st.checkbox('Save to Production Pipeline',key='show'):
            modelName = st.text_input("Input Model Name: ")
            if modelName:
                modelPipeline = ModelPipeline(modelName, aggregation_level, net_amount_quantiles, le_list, ct_model, model_obj)
                modelPipeline.saveModel(aggregation_level.replace(' ',''))
