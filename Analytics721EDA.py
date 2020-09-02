#Imports
from __future__ import division
%config IPCompleter.greedy=True
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
np.random.seed(1)

# Options for pandas and other packages
pd.options.display.max_columns = 50
pd.options.display.max_rows = 30
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
plt.rcParams["figure.figsize"] = (14, 10)
plt.style.use('ggplot')

%autosave 60

#Get local path
import os
cwd = os.getcwd()
cwd

################################################################################
#read files
#input_file = "filename.csv"
#df = pd.read_table(cwd + input_file, sep=',', error_bad_lines=False, encoding="ISO-8859-1")
#df = pd.read_table(input_file, sep='\t', error_bad_lines=False, encoding="ISO-8859-1")

#read json file in
#df = pd.read_json("train.json")
#df.head()

#df.info()
#df.head()

#other file Imports
#df = pd.read_excel("/Users/example.xlsx", sheet_name="Sheet1")

#write files
#df.to_csv('/Users/file.csv', sep=',', encoding='utf-8')

# DataBase Connection to Run EDA
import pyodbc
cnxn = pyodbc.connect(r'Driver={SQL Server};Server=XXXXXXXX;Database=XXXXXXXX;UID="XXXXXXX";Trusted_Connection=yes;')
sql = """SELECT TOP (10000) from XXXXXXXXX """

#df = pd.read_sql(sql,cnxn, parse_dates=['Date1', 'Date2'])

#drop collumns
#df = df.drop('Collumn_A', 1)

# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from IPython import get_ipython
ipython = get_ipython()

# autoreload extension
if 'autoreload' not in ipython.extension_manager.loaded:
    %load_ext autoreload

%autoreload 2

# Other Visualizations
#import plotly.plotly as py
#import plotly.graph_objs as go
#from plotly.offline import iplot, init_notebook_mode
#init_notebook_mode(connected=True)

#import cufflinks as cf
#cf.go_offline(connected=True)
#cf.set_config_file(theme='white')

#datetime functions
#df.dtypes # if already datetime64 you don't need to use to_datetime
#df['FirstActivityDate'] = pd.to_datetime(df['FirstActivityDate'])
#df['LastActivityDate'] = pd.to_datetime(df['LastActivityDate'])

#df['Daysbetween'] = (df['FirstActivityDate'] - df['LastActivityDate']).dt.days


#get names of categorical and numerical data columns
def divide_data_to_categorical_and_numerical(data):
        numerical_column_names=[]
        categorical_column_names=[]
        for column_name in data.columns:
            column= data[column_name] # column type is "pandas.core.series.Series"
            few=10
            few_elements=column.values[:few] # ten first elements
            is_categorical=[type(x) for x in few_elements].count(type("string"))==few
            #print("column: {0}. Categorical {1}".format(column_name,is_categorical))
            if(is_categorical):
                categorical_column_names.append(column_name)
            else:
                numerical_column_names.append(column_name)
        return categorical_column_names, numerical_column_names
#divide_data_to_categorical_and_numerical(df)


# From Kaggle's Avito Comp, use to make dataframe smaller in memory

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else: df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

#df_small=reduce_mem_usage(df)

###################################################################################
# set Target/Features for supervised
predictors = df.drop('Target', axis=1)
response = df['Target']
# you can use it like this is you need to

predictors = df.drop('Target', axis=1).columns
response = 'Target'

X = df[predictors]
y = df[response]


##################################################################################
#Standard EDA functions
#!pip install pandas_profiling
import pandas_profiling
pandas_profiling.ProfileReport(df)
profile = pandas_profiling.ProfileReport(df)
#profile.to_file(outputfile="Data profiling.html")#to save as HTML file


#import pandas as pd
#importing plotly and cufflinks in offline mode
import cufflinks as cf

import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
#plotly's interactive iplot
df.iplot()
#df.iplot(kind='scatter', filename='cufflinks/cf-simple-line')


def fillRates(df, pct=0.9):
    """
    pct: Fillrate threshhold below which a column will be listed.  This gives us a percetage of missing values
    """
    fillrates = df.count() / df.shape[0]
    print(fillrates[fillrates < pct].sort_values())
    return fillrates

def get_cols(df):
    #gives all the non id cols
    import re
    p = re.compile('(id|key|number)$', re.IGNORECASE)
    non_id_cols = [c for c in df.columns if not p.search(c)]
    return non_id_cols

def describe_continuous(df, include_zero=True, drop_right_tail=1):
    #include_zero if we include the 0's or not
    #drope_right_tail is how much of the right(extreme upper) values we include where 1=100%
    cols = df.select_dtypes(include=['float64', 'int64']).columns.values
    import re
    p = re.compile('(id|key|number)$', re.IGNORECASE)
    cols = [c for c in cols if not p.search(c)]
    print(df[cols].describe())
    plt.style.use('bmh')
    for c in cols:
        print(c)
        srs = df[c].dropna()
        if not include_zero:
            srs = srs[srs != 0]
        if drop_right_tail < 1:
            srs = srs[srs < srs.quantile(drop_right_tail)]
        val_count = len(df[c].drop_duplicates())
        if val_count > 50:
            val_count = val_count // 2 + 1
        elif val_count > 100:
            val_count = val_count // 4 + 1
            if val_count > 50:
                val_count = 50
        plt.hist(srs, histtype="stepfilled", bins=val_count, alpha=0.8)
        plt.show()

def describe_categorical(df):
    cols = df.select_dtypes(include=['object']).columns.values
    import re
    p = re.compile('(id|key|number)$', re.IGNORECASE)
    cols = [c for c in cols if not p.search(c)]
    for c in cols:
        print(c)
        print(df[c].value_counts())

##############################################################################


def describe_table(name):
    """
    name: string representing the name of the table.
    """
    df = pd.read_sql('select top 100000 * from dbo.' + name , cnxn)
    df.head()
    fillRates(df)
    describe_continuous(df)
    describe_categorical(df)



############################################################################
#Time Series EDA
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm

def tsplot(y, lags=None, figsize=(10, 8)):
    #import statsmodels.tsa.api as smt
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(1.5) for ax in [acf_ax, pacf_ax]]
    #sns.despine()
    plt.tight_layout()
    #plt.axvspan('2017-10-06', '2017-10-10', color='blue', alpha=0.5)
    return ts_ax, acf_ax, pacf_ax


# df needs to have index as date and be univariate(i think)
tsplot(df, lags=15)
plt.show()


# Create TS Features
def create_features(df, datecol):
    df[datecol+'_weekend'] = ((df[datecol].dt.dayofweek) // 5 == 1).astype(float)
    df[datecol+'_weekday'] = df[datecol].apply(lambda x: x.weekday()).astype(object)
    df[datecol+'_month']=df[datecol].dt.month.astype(object)
    df[datecol+'_day']=df[datecol].dt.day.astype(object)
    df[datecol+'_DayOfWeekNumb']=df[datecol].dt.dayofweek
    df[datecol+'_DayOfWeek']=df[datecol].dt.weekday_name
    df[datecol+'_year']=df[datecol].dt.year
    df[datecol+'_Quarter'] = df[datecol].dt.quarter
    df[datecol+'_hour'] = df[datecol].dt.hour
    df[datecol+'_dayofyear'] = df[datecol].dt.dayofyear
    df[datecol+'_weekofyear'] = df[datecol].dt.weekofyear


create_features(df1,'Calendar Date')

#add US Federal Holiday Calendar features from Pandas from here https://www.kaggle.com/robikscube/tutorial-time-series-forecasting-with-prophet
# or here
#https://stackoverflow.com/questions/29688899/pandas-checking-if-a-date-is-a-holiday-and-assigning-boolean-value

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

cal = calendar()
train_holidays = cal.holidays(start=df_train.index.min(), end=df_train.index.max())
test_holidays = cal.holidays(start=df_test.index.min(), end=df_test.index.max())
# Create a dataframe with holiday, ds columns
df['date'] = df.index.date
df['is_holiday'] = df.date.isin([d.date() for d in cal.holidays()])
holiday_df = df.loc[pjme['is_holiday']].reset_index().rename(columns={'Datetime':'ds'})
holiday_df['holiday'] = 'USFederalHoliday'
holiday_df = holiday_df.drop(['df_MW','date','is_holiday'], axis=1)
holiday_df.head()




def strc(df):
    print("INFORMATION ON DATAFRAME")
    print(df.info())
    print("STATISTICS ON DATAFRAME")
    print(df.describe())
    print("COUNT OF NULL VALUES")
    print(df.isnull().sum())
    print("TOP 5 ROWS")
    print(df.head())

strc(df)

#Summary by column including outliers and ratios
def summarize(Dataframe, DF_Name, df_col, n=3):
    queueList = str.split(DF_Name, ';')
    train_data=df_col.notnull().sum()
    zero_data=(df_col==0).sum()
    meanVol=df_col.mean()
    stdVol=df_col.std()
    #n=3, number of standard deviations
    outlier_count=((df_col < (meanVol - n*stdVol)) | (df_col > (meanVol + n*stdVol))).sum()
    outlier_ratio=outlier_count/Dataframe.shape[0]
    missing_count=df_col.isnull().sum()
    missing_ratio=missing_count/Dataframe.shape[0]
    result = queueList
    result.append(train_data)
    print('mean_Amount:')
    print(meanVol)
    print('std_Amount:')
    print(stdVol)
    print('zeros_in_data:')
    print(zero_data)
    print('Outliers_in_data:')
    print(outlier_count)
    print('Outliers_Ratio_in_data:')
    print(outlier_ratio)
    print('Missing_Count_in_data:')
    print(missing_count)
    print('Missing_Ratio_in_data:')
    print(missing_ratio)
    return result

summarize(df,'DataFrame_TestName',df['col_name'], n=3)

#time series plots, requires one of the columns to be date
def ts_plot(ts, plotTitle, col_name):
    mpl_fig = plt.figure()

    axes = mpl_fig.add_subplot(111)#top left
    #axes = mpl_fig.add_subplot(212)#top right
    #axes = mpl_fig.add_subplot(213)#bottom left
    #axes = mpl_fig.add_subplot(214)#bottom right
    ts.reindex(copy=False)

    # test that dates are sorted\
    date_series = ts["Date"]
    is_sorted = all(date_series[i] <= date_series[i+1] for i in range(len(date_series)-1))
    if not is_sorted:
        ts = ts.sort_values("Date")

    axes.plot(ts.Date, col_name)
    axes.set_title(plotTitle)
    axes.set_ylabel("Amount")
    axes.set_xlabel("Calender Date")
    axes.set_visible(True)
    axes.tick_params(axis='both', which='both', labelsize=7, labelbottom=True)
    return axes

#############################################################################
# OLS With and Without a constant
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

#df.corr(method = 'pearson')
# for pearson correlation for all numberic columns

#X = df[['Collumn by name', 'collumn A', 'Collumn B']]
#Collumns by Numbers instead
#X = df[[0,1,2,3]]
#y = df['Depedent Variable']
#X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model


# Note the difference in argument order
#model = sm.OLS(y, X).fit()
#predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
#model.summary()

############################################################################
#Logistic Model with SKlearn, statsmodels also has this
# Import logistic regression package
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, jaccard_similarity_score, roc_auc_score
import matplotlib.pyplot as plt

# You will need a Train test split here for Xs and Y
# logistic regression model
model = LogisticRegression()
#model = LogisticRegression(multi_class='multinomial',solver='lbfgs')
model = model.fit(X_train, y_train)
Y_test = model.predict(X_test)
matches = (Y_test == Y_train)
explained_variance = model.score(X_test,y_test)
print("R^2: {}".format(explained_variance))
print 'Correct N = %d' %(matches.sum())
print 'Total   N = %d' %(len(matches))
print "model accuracy is %.2f%%" % (matches.sum()*1.0/len(matches)*100)

print("Estimated coefficients for the logistic regression : {}".format(model.coef_))

# generate class probabilities
probs = model.predict_proba(X_test)
print(probs)
Y_test = model.predict(X_test)

ev = explained_variance_score(y_test, Y_test, multioutput='uniform_average')
# Best possible score is 1.0, lower values are worse.
print("Explained Variance Score: {}". format(ev))

mae = mean_absolute_error(y_test, Y_test, multioutput='uniform_average')
# MAE output is non-negative floating point. The best value is 0.0.
print("Mean Absolute Error: {}".format(mae))

mse = mean_squared_error(y_test, Y_test, multioutput='uniform_average')
# MAE output is non-negative floating point. The best value is 0.0.
print("Mean Squared Error: {}".format(mse))

r2 = r2_score(y_test, Y_test)
# R^2 (coefficient of determination) regression score function.
# Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always
# predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.
print("R - Squared value: {}".format(r2))

print('What percent of predictions are same: {}'.format(jaccard_similarity_score(y_test, Y_test)))

# Confusion Matrix
print(metrics.confusion_matrix(y_test, Y_test))
print(metrics.classification_report(y_test, Y_test))

actual = y_train
predictions = model.predict(X_train)
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("Area Under the curve is: {}".format(roc_auc))

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print(scores)
print(scores.mean())

####################################################################
#for more detailed example see
#http://www.statsmodels.org/devel/examples/notebooks/generated/interactions_anova.html
#simple ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
#anova_model=ols('continious_variable ~ grouping_variable', data=df).fit()
#aov_table=sm.stats.anova_lm(anova_model, typ=2)
#print(aov_table)
#esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
#print(esq_sm)#for TSS

#Dummy encoding(one hot)
def dummy_coding_for_vars(df, list_of_variables,  dummy_na=False, drop_first = False, prefix=None):
    if prefix==None:
        prefix = list_of_variables
    outputdata = pd.get_dummies(df, columns=list_of_variables, prefix= prefix, dummy_na=dummy_na, drop_first=drop_first)
    return outputdata

####################################################
#other model types Imports
#import importlib
##importlib.reload(module) #usually don't need thi
#from keras import backend as K
#from os import environ

# user defined function to change keras backend
def set_keras_backend(backend):
    if K.backend() != backend:
       environ['KERAS_BACKEND'] = backend
       importlib.reload(K)
       assert K.backend() == backend

# call the function with "theano" for other backends
#set_keras_backend("theano")

####################################################
#tensorflow without Keras
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

tf.enable_eager_execution()
np.random.seed(0)


#df= shuffle(df)#if you want to randomize your max_rows


processed_features = df[['Column1', 'Column2' '...']]
output_targets = df[['Column_DV1']]
# this is just the DV but it appears you can use multiple columns here or a construct

# train, val, test split
# We should use the best method possible here but I will give examples of one simple way
training_examples = processed_features[0:1000]
training_targets = output_targets[0:1000]

val_examples = processed_features[1000:1200]
val_targets = output_targets[1000:1200]

test_examples = processed_features[1200:1500]
test_targets = output_targets[1200:1500]

my_feature_columns = [tf.feature_column.numeric_column("Column1", '...')] # this is from our processed_features list
#
feat1=tf.feature_column.numeric_column('feat1')
feat2=tf.feature_column.numeric_column('feat2')# and so on ...

my_optimizer = tf.train.GradientDecentOptimizer(learning_rate=.01)
my_optimizer = tf.contrib.estimator.club_gradients_by_norm(my_optimizer, 5.0)
#https://www.tensorflow.org/api_docs/python/tf/train
# use the link for other otimizers
# tf.keras also has it own optimizers liek the one below,
#https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
#according to stack overflow you should not mix Keras and Tensorflow models as the two APIs don't go together

#for TF we not only need our feature columns and other parameters but we also need an optimizer mathmatical function
#canned optimizer
model = tf.estimator.LinearRegressor(feature_columns=my_feature_columns, optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9))
#custom optimizers
model = tf.estimator.LinearRegressor(feature_columns=my_feature_columns, optimizer=my_optimizer)
model = tf.estimator.DNNRegressor(feature_columns=my_feature_columns, hidden_units=[12,12], optimizer=my_optimizer)

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    #try num_epochs=1  too
    #convert pandas data into a dict of np arrays
    features={key:np.array(value) for key,value in dect(features).items()}

    #construct a dataset and configure batching/repeatign
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2gb limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    #shuffle data if specified
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    #return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels

#train model on training dataset
training = model.train(input_fn = lambda:my_input_fn(training_examples, training_targets), steps=1000)
#for DNN
training = model.train(input_fn = lambda:my_input_fn(training_examples, training_targets['Column_DV1'], batch_size=32), steps=1000)

#Evaluatingthe model using RSME
train_predictions = model.predict(input_fn = lambda:my_input_fn(training_examples, training_targets, num_epochs=1, suffle=False))
val_predictions = model.predict(input_fn = lambda:my_input_fn(val_examples, val_targets, num_epochs=1, suffle=False))
test_predictions = model.predict(input_fn = lambda:my_input_fn(test_examples, test_targets, num_epochs=1, suffle=False))

#format predicitons as Numpy array so we can calcluate error metrics
train_predictions = np.array([item['predictions'][0] for item in train_predictions])
val_predictions = np.array([item['predictions'][0] for item in val_predictions])
test_predictions = np.array([item['predictions'][0] for item in test_predictions])

#print MSE and RMSE

mean_squared_error = metrics.mean_squared_error(train_predictions, training_targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

mean_squared_error = metrics.mean_squared_error(val_predictions, val_targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Root Mean Squared Error (on validation data): %0.3f" % root_mean_squared_error)

mean_squared_error = metrics.mean_squared_error(test_predictions, test_targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Root Mean Squared Error (on test data): %0.3f" % root_mean_squared_error)

########################################################################
# Weigthed Absolute Percentage Error
def wape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / np.mean(y_true))) * 100


Error = wape(y_Validation, predicted_validation[:, 1])
Error

########################################################################
#other supervised metrics with numpy for continious variables

def weighted_absolute_percent_error(ground_truth, predictions):
    ground_truth, predictions = np.array(ground_truth), np.array(predictions)
    return(np.sum(np.abs(ground_truth-predictions))/np.sum(ground_truth))

def mean_squared_error(ground_truth, predictions):
    ground_truth, predictions = np.array(ground_truth), np.array(predictions)
    return(np.square(ground_truth-predictions).mean())

def root_mean_squared_error(ground_truth, predictions):
    ground_truth, predictions = np.array(ground_truth), np.array(predictions)
    return(np.sqrt(np.square(ground_truth-predictions).mean()))


###############################################################################
#using H2O for Modeling
#connection to H20
#!pip install h2o
import h2o
import time
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


# Connect to a cluster
#pip install h2o
h2o.init(nthreads=4, max_mem_size = "16g")
h2o.connect()

h2o.remove_all()



train_file = "example_train.csv"
test_file = "example_test.csv"

train = h2o.import_file(train_file)
test = h2o.import_file(test_file)

# To see a brief summary of the data, run the following command
train.describe()
test.describe()



h2o.cluster().shutdown()





# debug magic works well
%debug

import matplotlib.pyplot as plt
%matplotlib inline

#bargraph and add others here.
#fig = plt.figure(figsize=(8,6))
#df.groupby('Theme').text.count().plot.bar(ylim=0)
#plt.show()

#graphics functions

def categorical_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', verbose=True):

    '''

    Helper function that gives a quick summary of a given column of categorical data
    Arguments
    =========
    dataframe: pandas dataframe

    x: str. horizontal axis to plot the labels of categorical data, y would be the count
    y: str. vertical axis to plot the labels of categorical data, x would be the count
    hue: str. if you want to compare it another variable (usually the target variable)
    palette: array-like. Colour of the plot

    Returns
    =======
    Quick Stats of the data and also the count plot
    '''

    if x == None:

        column_interested = y

    else:

        column_interested = x

    series = dataframe[column_interested]

    print(series.describe())

    print('mode: ', series.mode())

    if verbose:

        print('='*80)

        print(series.value_counts())

    sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette=palette)

    plt.show()


#examples of function
c_palette = ['tab:blue', 'tab:orange']
#categorical_summarized(train_df, y = 'cat_var1', palette=c_palette)
#categorical_summarized(train_df, y = 'cat_var2', hue='cat_var1', palette=c_palette)
#looks liek these cat_vars can be either int64s or objects

def quantitative_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', ax=None, verbose=True, swarm=False):

    '''

    Helper function that gives a quick summary of quantattive data

    Arguments

    =========

    dataframe: pandas dataframe

    x: str. horizontal axis to plot the labels of categorical data (usually the target variable)
    y: str. vertical axis to plot the quantitative data
    hue: str. if you want to compare it another categorical variable (usually the target variable if x is another variable)
    palette: array-like. Colour of the plot
    swarm: if swarm is set to True, a swarm plot would be overlayed
    Returns
    =======
    Quick Stats of the data and also the box plot of the distribution

    '''

    series = dataframe[y]

    print(series.describe())

    print('mode: ', series.mode())

    if verbose:

        print('='*80)

        print(series.value_counts())


    sns.boxplot(x=x, y=y, hue=hue, data=dataframe, palette=palette, ax=ax)


    if swarm:

        sns.swarmplot(x=x, y=y, hue=hue, data=dataframe,

                      palette=palette, ax=ax)

    plt.show()


#quantitative_summarized(dataframe= train_df, y = 'float64_var1', palette=c_palette, verbose=False, swarm=True)
# this may work with numberic variables instead of floats64s as well
#quantitative_summarized(dataframe= train_df, y = 'float64_var1', x = 'cat_var1, palette=c_palette, verbose=False, swarm=True)
#our number variable float64_var1 graphed by group of cat_var1
#quantitative_summarized(dataframe= train_df, y = 'float64_var1', x = 'cat_var3', hue = 'cat_var4', palette=c_palette3, verbose=False, swarm=False)
#cat_var3 is an object in the example, and cat_var4 is an int64(categorical variable)




#############################################################################

# Current version of EDA functions Students post bachelors Level

import matplotlib.pyplot as plt

def fillRates(df, pct=0.9):
    """
    pct: Fillrate threshhold below which a column will be listed.  This gives us a percetage of missing values
    """
    fillrates = df.count() / df.shape[0]
    print(fillrates[fillrates < pct].sort_values())
    return fillrates

def get_cols(df):
    #gives all the non id cols
    import re
    p = re.compile('(id|key|number)$', re.IGNORECASE)
    non_id_cols = [c for c in df.columns if not p.search(c)]
    return non_id_cols

def describe_continuous(df, include_zero=True, drop_right_tail=1):
    #include_zero if we include the 0's or not
    #drope_right_tail is how much of the right(extreme upper) values we include where 1=100%
    cols = df.select_dtypes(include=['float64', 'int64']).columns.values
    import re
    p = re.compile('(id|key|number)$', re.IGNORECASE)
    cols = [c for c in cols if not p.search(c)]
    print(df[cols].describe())
    plt.style.use('bmh')
    for c in cols:
        print(c)
        srs = df[c].dropna()
        if not include_zero:
            srs = srs[srs != 0]
        if drop_right_tail < 1:
            srs = srs[srs < srs.quantile(drop_right_tail)]
        val_count = len(df[c].drop_duplicates())
        if val_count > 50:
            val_count = val_count // 2 + 1
        elif val_count > 100:
            val_count = val_count // 4 + 1
            if val_count > 50:
                val_count = 50
        plt.hist(srs, histtype="stepfilled", bins=val_count, alpha=0.8)
        plt.show()

def describe_categorical(df):
    cols = df.select_dtypes(include=['object']).columns.values
    import re
    p = re.compile('(id|key|number)$', re.IGNORECASE)
    cols = [c for c in cols if not p.search(c)]
    for c in cols:
        print(c)
        print(df[c].value_counts())

def describe_table(name):
    """
    name: string representing the name of the table.
    """
    df = psql.read_sql('select top 100000 * from dbo.' + name , cnxn)
    df.head()
    fillRates(df)
    describe_continuous(df)
    describe_categorical(df)




#Time Series EDA
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm

def tsplot(y, lags=None, figsize=(10, 8)):
    #import statsmodels.tsa.api as smt
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(1.5) for ax in [acf_ax, pacf_ax]]
    #sns.despine()
    plt.tight_layout()
    #plt.axvspan('2017-10-06', '2017-10-10', color='blue', alpha=0.5)
    return ts_ax, acf_ax, pacf_ax


# df needs to have index as date and be univariate(i think)
tsplot(df, lags=15)
plt.show()


# Create TS Features
def create_features(df, datecol):
    df[datecol+'_weekend'] = ((df[datecol].dt.dayofweek) // 5 == 1).astype(float)
    df[datecol+'_weekday'] = df[datecol].apply(lambda x: x.weekday()).astype(object)
    df[datecol+'_month']=df[datecol].dt.month.astype(object)
    df[datecol+'_day']=df[datecol].dt.day.astype(object)
    df[datecol+'_DayOfWeekNumb']=df[datecol].dt.dayofweek
    df[datecol+'_DayOfWeek']=df[datecol].dt.weekday_name
    df[datecol+'_year']=df[datecol].dt.year
    df[datecol+'_Quarter'] = df[datecol].dt.quarter


create_features(df1,'Calendar Date')

def strc(df):
    print("INFORMATION ON DATAFRAME")
    print(df.info())
    print("STATISTICS ON DATAFRAME")
    print(df.describe())
    print("COUNT OF NULL VALUES")
    print(df.isnull().sum())
    print("TOP 5 ROWS")
    print(df.head())

strc(df)

# list all days with missing Values
nans = lambda df: df[df.isnull().any(axis=1)]

missing=nans(df)
missing.index


# relace all commas or whaever you want with Blanks
	# to pull cols not equal to...
	#cols = [col for col in  df.columns if col != 'Roll Up']
cols = df.columns.get_values()
cols=cols.tolist()
cols

		# remove the ones you don't want this filtering applied to
cols.remove('Calendar Date')
cols.remove('Calendar Date_DayOfWeek')
cols

def func(x):
    x = x.astype(str)
    x = x.str.replace(',', '')
    x = x.astype(float)
    return x

df[cols] = df[cols].apply(func)




# Weigthed Absolute Percentage Error
def wape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / np.mean(y_true))) * 100


Error = wape(y_Validation, predicted_validation[:, 1])
Error
