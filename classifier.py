# To run the script type python classifier.py or python3 classifier.py (this depends on your OS)
# The script expects the dataset to have been extracted in a folder called data in the same directory
# as classifier.py

# The script requires the latest versions of pandas and sklearn to run
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.svm import LinearSVC

# Init filepaths for the dataset CSV files
ASSESSMENTS_FPATH = os.path.join(os.getcwd(), 'data', 'assessments.csv')
COURSES_FPATH = os.path.join(os.getcwd(), 'data', 'courses.csv')
STUDENT_ASSESS_FPATH = os.path.join(
    os.getcwd(), 'data', 'studentAssessment.csv')
STUDENT_INFO_FPATH = os.path.join(os.getcwd(), 'data', 'studentInfo.csv')
STUDENT_REG_FPATH = os.path.join(
    os.getcwd(), 'data', 'studentRegistration.csv')
STUDENT_VLE_FPATH = os.path.join(os.getcwd(), 'data', 'studentVle.csv')
VLE_FPATH = os.path.join(os.getcwd(), 'data', 'vle.csv')


# function ordinally encodes the final results
# ordered ascending from withdrawn 0 to Distinction 3
def ordinal_final_result(student_info):
    mod_res = student_info.sort_values(by=['id_student'])
    mod_res = mod_res[['final_result']]
    mod_res.loc[mod_res['final_result'] == 'Withdrawn', 'final_result'] = 0
    mod_res.loc[mod_res['final_result'] == 'Fail', 'final_result'] = 1
    mod_res.loc[mod_res['final_result'] == 'Pass', 'final_result'] = 2
    mod_res.loc[mod_res['final_result'] == 'Distinction', 'final_result'] = 3
    return mod_res


# function calculates the spearman rank correlation coefficient
# between final result and students total assessment mark
def assess_mark_corr(student_info):
    # open dataset files and load the data
    mod_res = ordinal_final_result(student_info)
    stud_assess = pd.read_csv(STUDENT_ASSESS_FPATH)
    assess = pd.read_csv(ASSESSMENTS_FPATH)
    # join assessment and studen assessment tables on id_assessment
    stud_assess = pd.merge(stud_assess, assess, on='id_assessment')
    # isolate relevant student info columns for the correlation calc
    features = student_info[['id_student', 'code_module', 'code_presentation']]
    # join tables on student id and module codes
    stud_assess_merged = pd.merge(features, stud_assess, on=[
        'id_student', 'code_module', 'code_presentation'], how='left')
    # assume any missing entries are 0 scores
    stud_assess_merged = stud_assess_merged.fillna(0)
    # total up student assessment marks for each module and student
    stud_assess_merged = stud_assess_merged.groupby(
        ['id_student', 'code_module', 'code_presentation'])['score'].sum()
    stud_assess_merged.fillna(0, inplace=True)
    # calculate spearmans rank correlation coefficient
    res = stats.spearmanr(stud_assess_merged, mod_res)
    print('Correlation between total assessment mark for module and module' +
          ' result')
    print('Coefficient=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


# Calculates spearmans rank correlation coefficient
# between studen reg data and final result
def reg_data_corr(student_info):
    mod_res = ordinal_final_result(student_info)
    # load relevant CSV files
    stud_reg = pd.read_csv(STUDENT_REG_FPATH)
    # join tables on student id and module codes
    features = pd.merge(student_info, stud_reg, on=[
        'id_student', 'code_module', 'code_presentation'])
    features = features.sort_values(by=['id_student'])
    features.fillna(0, inplace=True)
    # calculate correlation coefficient
    res = stats.spearmanr(features[['date_registration']], mod_res)
    print('Correlation between registration date and module result')
    print('Coefficient=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


# calulates spearmans rank correlation correlation
# between unregistration data and final result
def unreg_data_corr(student_info):
    mod_res = ordinal_final_result(student_info)
    # load relevant CSV files
    stud_reg = pd.read_csv(STUDENT_REG_FPATH)
    # join tables on student id and module codes
    features = pd.merge(student_info, stud_reg, on=[
        'id_student', 'code_module', 'code_presentation'])
    features = features.sort_values(by=['id_student'])
    features.fillna(0, inplace=True)
    # calculate correlation coefficient
    res = stats.spearmanr(features[['date_unregistration']], mod_res)
    print('Correlation between de-registration date and module result')
    print('Coefficient=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


# calculates spearmans rank correlation coefficient
# between prev module attempts and final result
def prev_attempts_corr(student_info):
    mod_res = ordinal_final_result(student_info)
    sort_data = student_info.sort_values(by=['id_student'])
    # calculate correlation coefficient
    res = stats.spearmanr(sort_data[['num_of_prev_attempts']], mod_res)
    print('Correlation between previous module attempts and module result')
    print('Coefficient=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


# calculates spearmans rank correlation coefficient
# between module credits and final result
def stud_credits_corr(student_info):
    mod_res = ordinal_final_result(student_info)
    sort_data = student_info.sort_values(by=['id_student'])
    # calculate correlation coefficient
    res = stats.spearmanr(sort_data[['studied_credits']], mod_res)
    print('Correlation between studied credits for module and module result')
    print('Coefficient=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


# calulates spearmans rank correlation coefficient
# between course length and final_result
def course_length_corr(student_info):
    mod_res = ordinal_final_result(student_info)
    # load relevant CSV files
    courses = pd.read_csv(COURSES_FPATH)
    # join tables on module codes
    features = pd.merge(student_info, courses, on=[
        'code_presentation', 'code_module'])
    features = features.sort_values(by=['id_student'])
    # calculate correlation coefficient
    res = stats.spearmanr(
        features[['module_presentation_length']], mod_res)
    print('Correlation between course length and module result')
    print('Coefficient=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


# performs a chi squared association test beween
# student gender and final result
def gender_association(student_info):
    freq = student_info[['gender', 'final_result']]
    # create contingency table for test
    freq = freq.groupby(['gender', 'final_result']).size().unstack()
    # perform test
    res = stats.chi2_contingency(freq)
    print('Association between gender and module result')
    print('Test stat=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


# performs chi squared association test between
# student imd band and final result
def imd_association(student_info):
    freq = student_info[['imd_band', 'final_result']]
    # create contingency table
    freq = freq.groupby(['imd_band', 'final_result']).size().unstack()
    # perform test
    res = stats.chi2_contingency(freq)
    print('Association between imd and module result')
    print('Test stat=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


# performs chi squared test between
# students highest level of education and final result
def higher_ed_association(student_info):
    freq = student_info[['highest_education', 'final_result']]
    # creat contingency table
    freq = freq.groupby(['highest_education', 'final_result']).size().unstack()
    # perform test
    res = stats.chi2_contingency(freq)
    print('Association between highest education level and module result')
    print('Test stat=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


# performs chi squared test between
# students home region and final result
def region_association(student_info):
    freq = student_info[['region', 'final_result']]
    # creat contingency table
    freq = student_info.groupby(['region', 'final_result']).size().unstack()
    # perform test 
    res = stats.chi2_contingency(freq)
    print('Association between student region and module result')
    print('Test stat=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


# performs chi squared test between
# student disability classification and final result
def disability_association(student_info):
    freq = student_info[['disability', 'final_result']]
    # create contingency table
    freq = student_info.groupby(
        ['disability', 'final_result']).size().unstack()
    # perform test
    res = stats.chi2_contingency(freq)
    print('Association between disability and module result')
    print('Test stat=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


# perform chi squared test between
# students age band and final_result
def age_band_association(student_info):
    freq = student_info[['age_band', 'final_result']]
    # create contingency table
    freq = student_info.groupby(
        ['age_band', 'final_result']).size().unstack()
    # perform test
    res = stats.chi2_contingency(freq)
    print('Association between age and module result')
    print('Test stat=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


# joines tables with chosen features for model training
def join_tables(student_info):
    # open relevant CSV files
    stud_reg = pd.read_csv(STUDENT_REG_FPATH)
    courses = pd.read_csv(COURSES_FPATH)
    # join student reg table and course table
    merged = pd.merge(student_info, courses, on=[
        'code_module', 'code_presentation'])
    # join result with student info table
    merged = pd.merge(merged, stud_reg, on=[
        'id_student', 'code_presentation', 'code_module'])
    return merged


# split data set into features table and labels table
def transform_for_model(data):
    features = data.drop('final_result', axis=1)
    labels = list(data['final_result'])
    # return tuple
    return (features, labels)


# creates a data pipe to transform feature table into acceptable input
# for the models
def create_data_pipe(features):
    # isolate categorical features
    cat_features = features[['gender', 'highest_education',
                             'age_band', 'region', 'disability']]
    # isolate numerical features
    num_features = features[['num_of_prev_attempts', 'studied_credits',
                             'module_presentation_length',
                             'date_registration']]
    # isolate 2 features that need some special processing
    imd_features = features[['imd_band']]
    unreg_feature = features[['date_unregistration']]
    # numerical features have missin values replaced with median
    # and are scaled about 0 to have a near normal distribution
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # unreg date has unknowns replaced by a constant value instead
    # as this obviously signifies withdrawal. Also scaled
    unreg_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=999)),
        ('scaler', StandardScaler())
    ])
    # categorical values are replaced by most frequent and one hot encoded
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('enc', OneHotEncoder())
    ])
    # imd is same as above but ordinally encoded because the categories
    # are clearly ordered
    imd_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('enc', OrdinalEncoder())
    ])
    # Merge transformed features back into one table
    comp_pipe = ColumnTransformer([
        ('num', num_pipeline, list(num_features)),
        ('reg', unreg_pipeline, list(unreg_feature)),
        ('cat', cat_pipeline, list(cat_features)),
        ('imd', imd_pipe, list(imd_features))
    ])
    return comp_pipe


def process_data(train_set):
    # data analysis
    assess_mark_corr(train_set)
    prev_attempts_corr(train_set)
    stud_credits_corr(train_set)
    course_length_corr(train_set)
    reg_data_corr(train_set)
    unreg_data_corr(train_set)
    gender_association(train_set)
    imd_association(train_set)
    higher_ed_association(train_set)
    region_association(train_set)
    disability_association(train_set)
    age_band_association(train_set)


# train the random forrest classifier on a training set
def train_forest(train_set):
    # get features and labels from training set
    joined = join_tables(train_set)
    features, labels = transform_for_model(joined)
    # get a data pipe
    data_pipe = create_data_pipe(features)
    for_class = RandomForestClassifier()
    # parameters for grid search hyperparamter tuning
    # only optimal values are here as I have already carried out the tuning
    params = [
        {'n_estimators': [50], 'max_features':[5]},
    ]
    # create grid search using micro averaged precision as scoring measure
    g_search = GridSearchCV(for_class, params, cv=10, scoring=[
                            'precision_micro', 'recall_micro'],
                            return_train_score=True, refit='precision_micro')
    # add data pipe and grid search to pipe so model will transform data
    # before predicting
    forest_pipe = Pipeline([
        ('data_pipe', data_pipe),
        ('grid', g_search)
    ])
    # train the model on the training set
    forest_pipe.fit(features, labels)
    # print best hyperparamters
    print(forest_pipe['grid'].best_params_)
    results = forest_pipe['grid'].cv_results_
    res_tuple = zip(results['mean_test_precision_micro'],
                    results['mean_test_recall_micro'], results['params'])
    # print micro averaged precision and recall scores for each set of hyperparamters
    for prec, rec, params in res_tuple:
        print('Prec=', prec, 'Recall=', rec, 'Params=', params)
    # save a local copy of the saved model
    save_model(forest_pipe, 'forest_pipe.pkl')
    return forest_pipe


# train support vector classifier model on training set
def train_svc(train_set):
    joined = join_tables(train_set)
    # get features and labels from train set
    features, labels = transform_for_model(joined)
    # get a data pipe to transform the featues
    data_pipe = create_data_pipe(features)
    svc = LinearSVC()
    # hyperparamters for grid search, only optimal values here as
    # hyperparamter tuning has already been done
    params = [
        {'C': [0.1], 'dual':[False]},
    ]
    # grid search with micro averaged precision as scoring metric
    g_search = GridSearchCV(svc, params, cv=10, scoring=[
                            'precision_micro', 'recall_micro'],
                            return_train_score=True, refit='precision_micro')
    # combine data pipe and grid search to create combined model pipe
    svc_pipe = Pipeline([
        ('data_pipe', data_pipe),
        ('grid', g_search)
    ])
    # train model
    svc_pipe.fit(features, labels)
    # print best hyperparamters
    print(svc_pipe['grid'].best_params_)
    results = svc_pipe['grid'].cv_results_
    res_tuple = zip(results['mean_test_precision_micro'],
                    results['mean_test_recall_micro'], results['params'])
    # print results for all hyperparamter sets tested
    for prec, rec, params in res_tuple:
        print('Prec=', prec, 'Recall=', rec, 'Params=', params)
    # save local copy of the model
    save_model(svc_pipe, 'svc_pipe.pkl')
    return svc_pipe


# test model on test set
def test_model(model, test_data, name):
    joined = join_tables(test_data)
    # get features and labels from test set
    features, labels = transform_for_model(joined)
    # get list of predictions from model
    preds = model.predict(features)
    # score the resulting predictions using the micro averaged precision as the metric
    precision = precision_score(labels, preds, average='micro')
    # calculate accuracy metric (this is actually the same as micro averaged precision)
    accuracy = accuracy_score(labels, preds)
    # print text based confusion_matrix
    conf = confusion_matrix(labels, preds)
    # plot graphical confusion_matrix
    conf_plot = plot_confusion_matrix(model, features, labels, normalize=None,
                                      values_format='.9g', cmap=plt.cm.Blues)
    conf_plot.ax_.set_title(name)
    # print scores and confusion_matrix
    print('Precision=', precision, '\nAccuracy=', accuracy)
    print('Confusion Matrix:')
    print(conf)


# save a model to a local file
def save_model(model, name):
    joblib.dump(model, name)


# load a model from a local file
def load_model(name):
    model = joblib.load(name)
    return model


def main():
    # load student_info
    student_info = pd.read_csv(STUDENT_INFO_FPATH)
    # split data into train and test set 80, 20 split using stratified sampling on highest_education
    train_set, test_set = train_test_split(
        student_info, test_size=0.2, random_state=42,
        stratify=student_info['highest_education'])
    # perform data analysis
    process_data(train_set)
    # train random forest 
    rand_forest = train_forest(train_set)
    # test random forest
    test_model(rand_forest, test_set, 'Forest')
    # train support vector classifier
    svc = train_svc(train_set)
    # test support vector classifier
    test_model(svc, test_set, 'SVC')
    plt.show()


main()
