import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import math
import pickle
from sklearn.ensemble import IsolationForest
from collections import Counter
from PIL import Image
from prince import MCA
from dataclasses import dataclass
from .models.AutoencoderV1 import Autoencoder
from .global_config import *
import os
import scipy.stats as ss

from configparser import ConfigParser

parser = ConfigParser()
parser.read('settings.ini')

defaultModelId = int(parser.get('AnomalyDetection', 'defaultModelId'))

print ("CWD : ",os.getcwd())
basePath = os.getcwd()

def getSeverity(x):
    return (np.abs((ss.rankdata(x,method='average')/(len(x)))-0.5)*2)**2

def getSeverity2(x):
    return (ss.rankdata(x,method='average')/(len(x)))**2

def app():

    st.header("Anomaly Detection")
    st.write("""## Statistic Methods to Detect the Outliers

* Global Outliers: These will be the outliers we can obtain using the whole data. They could be mild or severe.
    * Severe outliers: We use a z-distribution based method to detect the anomalies.
    * Mild Outliers: We use a quantile based distribution method to detect them
* Local Outliers: These will be the outlier we can obtain using every subgroup of data that can be grouped under the same supplier/cost center/journal
    * Severe Outlier: We use a z-distribution based method to detect the anomalies.
    * Mild Outliers: We use a propreitary technique based on the bollinger bands to detect them. We treat every record as a time series entity and use all the information in the entry to generate multiple time series, each with distinctive characteristics
* Isolated Outliers: Machine Learning based approach to get the anomalies based on distribution of every feature in the hyperspace

## DeepLearning Solution

* We can treat our data to be semi-supervised using the statistical outliers obtained above as labels. That way we can allow analysts to play around with hyper-parameter tuning and also tuning input/semi-supervised criterion ## To Do
    """)


    ######################################
    ## Sidebar
    ######################################
    # Input your csv
    st.markdown("""<style>
    .sidebar .sidebar-content {
    background-color: #3090C7;
    background-image: linear-gradient(#3090C7,#3090C7);
    color: white; }
    </style>""", unsafe_allow_html=True)

    st.sidebar.header('Upload your CSV data')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    print("*"*100)
    print(uploaded_file)

    st.sidebar.markdown("""
    [Example CSV input file]
    ({}/easyInput/Apr21Data.csv)
    """.format(basePath))

    ######################################
    # Main panel
    ######################################
    st.subheader('Dataset')

    @dataclass(unsafe_hash=True)
    class FileInfo:
        name: str
        size: float

    def convert_size(size_bytes):
       if size_bytes == 0:
           return "0B"
       size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
       i = int(math.floor(math.log(size_bytes, 1024)))
       p = math.pow(1024, i)
       s = round(size_bytes / p, 2)
       return "%s %s" % (s, size_name[i])


    def getAnomaliesReport(dfRaw, modelName, modelPath):
        df = dfRaw.copy()

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

        numRows = len(df)

        def getStatisticslOutliers(df, featureColumns, modelName, modelPath):
            with open("easyInput/labelEncoders.pkl",'rb') as f:
                le_list = pickle.load(f)
                f.close()

            for col in le_list.keys():
                print(col)
                colSeries = df[col].astype(str).astype("category")
                le = le_list[col]
                df[col]=le.transform(colSeries)

            left_hinge1, right_hinge1 = LEFT_HINGE1, RIGHT_HINGE1
            left_hinge2, right_hinge2 = LEFT_HINGE2, RIGHT_HINGE2
            mean_amt = df.net_amount.mean()
            std_amt = df.net_amount.std()
            left_hinge1 = min(mean_amt-(2*std_amt), left_hinge1)
            right_hinge1 = max(mean_amt+(2*std_amt), right_hinge1)
            median_amt = df.net_amount.quantile(0.5)
            iqr_amt = df.net_amount.quantile(0.95) - df.net_amount.quantile(0.05) # Usually 0.25 to 0.75
            left_hinge2 = min(median_amt-(1.5*iqr_amt), left_hinge2)
            right_hinge2 = max(median_amt+(1.5*iqr_amt), right_hinge2)

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

            global_outlier_serious = df.net_amount.apply(lambda x: 4 if (x<left_hinge1) or (x>right_hinge1) else 0)
            global_outlier_mild = df.net_amount.apply(lambda x: 2 if (x<left_hinge2) or (x>right_hinge2) else 0)
            local_outlier_serious = ((df.net_amount<left_hinges2) | (df.net_amount>right_hinges2))*3

            local_outlier_mild = ((df.net_amount.fillna(0)<left_hinges1) | (df.net_amount.fillna(0)>right_hinges1) | (df.net_amount.fillna(0)<left_hinges3) | (df.net_amount.fillna(0)>right_hinges3))*1
            with open("easyInput/isolationForestModel.pkl", "rb") as f:
                i_forest = pickle.load(f)
                f.close()

            anomaly_score = i_forest.decision_function(df.loc[:,featureColumns+otherReqColumns])
            anomaly_threshold = None
            for thresh,cnt in Counter(anomaly_score).most_common():
                if thresh>0.1:
                    anomaly_threshold = thresh
            outliers_isoForest = (anomaly_score>=anomaly_threshold)*5

            label = np.max([global_outlier_serious, global_outlier_mild, local_outlier_serious, local_outlier_mild, outliers_isoForest], axis=0)
            global_outlier_severity_score=getSeverity(df.net_amount)
            local_outlier_severity_score = df.groupby(featureColumns)['net_amount'].transform(getSeverity)
            isoForest_severity_score = getSeverity2(anomaly_score)

            num_outlier_categories = 1+global_outlier_serious+global_outlier_mild+local_outlier_serious+local_outlier_mild+outliers_isoForest
            overall_outlier_severity = ((1 - ((1-global_outlier_severity_score)*(1-local_outlier_severity_score)*(1-isoForest_severity_score)*(1-np.maximum(isoForest_severity_score,np.maximum(global_outlier_severity_score,local_outlier_severity_score))))**(1/4)))**(1/num_outlier_categories)


            return df, label, global_outlier_severity_score, local_outlier_severity_score, isoForest_severity_score, overall_outlier_severity

        df, label, global_outlier_severity_score, local_outlier_severity_score, isoForest_severity_score, overall_outlier_severity = getStatisticslOutliers(df, featureColumns, modelName, modelPath)
        outlierCodes = ['No Outlier','Local Outlier-Mild', 'Global Outlier-Mild','Local Outlier-Serious','Global Outlier-Serious','Isolated Characteristic Outlier']
        st.write("### Statistical Outliers In Data:")
        st.write("Total Number of transactions : {}".format(numRows))
        for k,v in dict(zip(*np.unique(label, return_counts=True))).items():
            st.write("* {} ({}) : {}".format(outlierCodes[k],k,v))
        st.write ("Total Num Outliers : {}".format((label>0).sum()))
        st.write ("Ratio Outliers : {}%".format(round((label>0).mean()*100,2)))
        dfRaw['outlierLabel'] = label
        dfRaw['outlierType'] = dfRaw['outlierLabel'].apply(lambda x: outlierCodes[x])
        dfRaw['global_outlier_severity_score']=global_outlier_severity_score
        dfRaw['local_outlier_severity_score']=local_outlier_severity_score
        dfRaw['isoForest_severity_score']=isoForest_severity_score
        dfRaw['overall_outlier_severity']=overall_outlier_severity
        st.write(dfRaw.loc[(label>0),:])

        image1 = Image.open('{}/easyInput/featuresGlobalImpact.png'.format(basePath))
        st.image(image1, caption='Global Data Features Impact - Outlier Detection')
        image2 = Image.open('{}/easyInput/featuresImpactCurr.png'.format(basePath))
        st.image(image2, caption='Features Impact in Uploaded Data - Outlier Detection')


        with open("easyInput/ohe.pkl","rb") as f:
            ohe = pickle.load(f)
            f.close()

        catDf = df.loc[:, featureColumns+otherReqColumns].copy()
        q = pd.read_pickle("easyInput/net_amount_quantiles.pkl")
        catDf['net_amount'] = pd.cut(catDf['net_amount'], bins=list(q), labels = list(range(len(q)-1)))
        for col in catDf.columns:
            catDf[col] = catDf[col].astype("category")
        allData = ohe.transform(catDf)
        if modelName == 'Auto Encoder V1-RecordLevel':

            model = Autoencoder(699)
            model.autoencoder.load_weights(modelPath)
            modelPredsRaw = model.predict(allData)

            modelPred = list()
            for actual, pred in zip(allData, modelPredsRaw):
                modelPred.append((actual*pred).sum()<0.55)
        else:
            with open(modelPath, 'rb') as modelFile:
                modelPipeline = pickle.load(modelFile)
                modelFile.close()
            testDf = dfRaw[featureColumns+otherReqColumns].copy()
            for col, le_model in modelPipeline.label_encoders.items():
                print("LE : ", col)
                testDf[col] = le_model.transform(testDf[col])
                testDf[col] = testDf[col].astype('category')
            if modelPipeline.column_transformers is not None:
                # testDf = np.hstack((ct_model.named_transformers_['one_hot_encoder'].inverse_transform(testDf[:,:-1]), testDf[:,-1:]))
                testDf = modelPipeline.column_transformers.transform(testDf)
            model = modelPipeline.model
            modelPredsRaw = model.predict(testDf)
            modelPred = [i==1 for i in modelPredsRaw]
        numAnomalousTxn = len([i for i in modelPred if i])
        st.write("### Outliers In Data - Using Model:")
        st.write("Total Number of transactions : {}".format(numRows))
        st.write("Number of Anomalous Transactions : {} ({}%)".format(numAnomalousTxn, round(numAnomalousTxn*100/numRows,2)))
        st.write(dfRaw.loc[modelPred,:])

        st.write("### Compare Outliers:")

        mca = MCA()
        mcaFeatures = mca.fit_transform(catDf)
        mcaFeatures.columns=['x','y']
        plotlyDf = mcaFeatures.copy()
        plotlyDf['statPred'] = label
        plotlyDf['modelPred'] = ['Anomalous Txn' if i else 'No Outlier' for i in modelPred]
        plotlyDf['statPred'] = plotlyDf['statPred'].apply(lambda x: outlierCodes[x])

        crossTabDf = plotlyDf.groupby(['statPred','modelPred']).size().reset_index().pivot(columns='modelPred', index='statPred').fillna(0).astype(int)
        crossTabDf.columns = crossTabDf.columns.levels[1]
        st.write(crossTabDf)

        figure1 = px.scatter(plotlyDf, x='x',y='y',color='statPred',width=1000, height=600, title='Statistical Outliers')
        figure1.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        st.plotly_chart(figure1)

        figure2 = px.scatter(plotlyDf, x='x',y='y',color='modelPred',width=1000, height=600, title='Outliers - Using Model')
        figure2.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        st.plotly_chart(figure2)
        return dfRaw

    def getIncidentReport(dfRaw):
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
        outlierCodes = ['No Outlier','Local Outlier-Mild', 'Global Outlier-Mild','Local Outlier-Serious','Global Outlier-Serious','Isolated Characteristic Outlier']
        dfRaw['outlierLabel'] = dfRaw['outlierLabel'].apply(lambda x: outlierCodes[x])

        incidentsAgg = dfRaw.groupby("outlierLabel").agg({"key_asof":'count', "net_amount":lambda x: int(np.abs(x).sum())}).reset_index().rename(columns={'key_asof':'NumTransactions', 'net_amount':'TotalTxnAmount'}).sort_values(['TotalTxnAmount','NumTransactions'])
        incidentsAgg['avgTxnAmount'] = (incidentsAgg['TotalTxnAmount']/incidentsAgg['NumTransactions']).astype('int')
        st.subheader("Outlier Incident Analysis")
        st.write(incidentsAgg)

        figure3 = px.pie(incidentsAgg, values='NumTransactions',names='outlierLabel',title='Number of Transactions by Outlier Type')
        st.plotly_chart(figure3)

        figure4 = px.pie(incidentsAgg, values='TotalTxnAmount',names='outlierLabel',title='Transaction Value/Amount by Outlier Type')
        st.plotly_chart(figure4)

        totalTxns = dict()
        for col in featureColumns:
            t1 = dfRaw.groupby(col).agg({"key_asof":'count', "net_amount":lambda x: np.abs(x).sum()}).reset_index().rename(columns={'key_asof':'NumTransactions', 'net_amount':'TotalTxnAmount'})
            t2 = dfRaw[dfRaw.outlierLabel!='No Outlier'].groupby(col).agg({"key_asof":'count', "net_amount":lambda x: np.abs(x).sum()}).reset_index().rename(columns={'key_asof':'AnomalousTransactions', 'net_amount':'AnomalousTxnAmount'})
            t3 = pd.merge(t1,t2,on=col, how='left').fillna(0)
            t3['AnomalousTxnPct'] = np.round(t3['AnomalousTransactions']*100/t3['NumTransactions'],2)
            t3['AnomalousTxnAmtPct'] = np.round(t3['AnomalousTxnAmount']*100/t3['TotalTxnAmount'],2)

            totalTxns[col] = t3.sort_values(['AnomalousTxnAmtPct','AnomalousTxnPct'], ascending=False)
            with st.expander("By {}".format(col), expanded=False):
                st.write(totalTxns[col])
            #st.session_state['incidentReport_'+col] = t3.sort_values(['AnomalousTxnAmtPct','AnomalousTxnPct'], ascending=False)
        # return totalTxns



        # if 'featureOutlierSummaryCol' not in st.session_state:
        #     st.session_state['featureOutlierSummaryCol'] = 'company_ref_id'
        # def updateFeatureOutlierSummaryOut():
        #     # st.write(totalTxns[st.session_state.featureOutlierSummaryCol])
        #     print ("Option Selected : ", st.session_state.featureOutlierSummaryCol)
        #
        # with st.form(key='featureOutlierSummaryForm'):
        #     option = st.selectbox('Select the Feature to get outlier summary',featureColumns, key='featureOutlierSummaryCol')
        #     submit_button = st.form_submit_button(label='Submit', on_click=updateFeatureOutlierSummaryOut)





    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        model_load_option = st.checkbox(
            "Load pre-Trained Model", True
        )
        if model_load_option:
            allModels = list()
            modelPaths = dict()
            for modelLevel in os.listdir("easyInput/modelPipelines/"):
                allFiles = os.listdir("easyInput/modelPipelines/"+modelLevel)
                for fn in allFiles:
                    modelName = MODELNAMES.get(fn, '{}-{}'.format(fn,modelLevel))
                    allModels.append(modelName)
                    modelPaths[modelName]="easyInput/modelPipelines/{}/{}".format(modelLevel, fn)

            if 'modelId' not in st.session_state:
                st.session_state['modelId'] = allModels[0]
            # def updateModelId():
            #     st.session_state['modelId']= allModels[allModels.index(st.session_state['modelId'])+1]
            selectedModel = st.selectbox("Select a pre-trained Model: ", options=allModels, key='modelId')

            dfRaw = getAnomaliesReport(df, selectedModel, modelPaths[selectedModel])
            getIncidentReport(dfRaw)
    else:
        st.info('Upload the data in csv format (comma separated)')
        # if 'exampleData' not in st.session_state:
        #     st.session_state['exampleData']=False
        # def on_click_example():
        #     useExample=True
        useExample = st.button('Press to use Example Dataset', disabled=False, key='exampleData')
        if useExample:
            df = pd.read_csv(
                '{}/easyInput/Apr21Data.csv'.format(basePath))
            st.markdown('The **Apr21Data.csv** dataset is used as the example.')
            if 'preTrained' not in st.session_state:
                st.session_state['preTrained']=True
            model_load_option = st.checkbox(
                "Load pre-Trained Model", key='preTrained'
            )
            if model_load_option:
                allModels = list()
                modelPaths = dict()
                for modelLevel in os.listdir("easyInput/modelPipelines/"):
                    allFiles = os.listdir("easyInput/modelPipelines/"+modelLevel)
                    for fn in allFiles:
                        modelName = MODELNAMES.get(fn, '{}-{}'.format(fn,modelLevel))
                        allModels.append(modelName)
                        modelPaths[modelName]="easyInput/modelPipelines/{}/{}".format(modelLevel, fn)

                if 'modelId' not in st.session_state:
                    try:
                        st.session_state['modelId'] = allModels[0] if defaultModelId is None else allModels[defaultModelId]
                    except:
                        st.session_state['modelId'] = allModels[0]

                def updateModelId():
                    print ("ModelID NOW :  ", st.session_state['modelId'])
                    defaultModelId = allModels.index(st.session_state['modelId'])
                    parser.set('AnomalyDetection', 'defaultModelId', str(defaultModelId))
                    with open('settings.ini','w') as settingsFile:
                        parser.write(settingsFile)
                        settingsFile.close()

                print ("NOWWWWWW: ", st.session_state['modelId'])
                selectedModel = st.selectbox("Select a pre-trained Model: ", options=allModels, key='modelId', index=allModels.index(st.session_state['modelId']), on_change=updateModelId)

                dfRaw = getAnomaliesReport(df, selectedModel, modelPaths[selectedModel])
                getIncidentReport(dfRaw)
