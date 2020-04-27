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


ASSESSMENTS_FPATH = os.path.join(os.getcwd(), 'data', 'assessments.csv')
COURSES_FPATH = os.path.join(os.getcwd(), 'data', 'courses.csv')
STUDENT_ASSESS_FPATH = os.path.join(
    os.getcwd(), 'data', 'studentAssessment.csv')
STUDENT_INFO_FPATH = os.path.join(os.getcwd(), 'data', 'studentInfo.csv')
STUDENT_REG_FPATH = os.path.join(
    os.getcwd(), 'data', 'studentRegistration.csv')
STUDENT_VLE_FPATH = os.path.join(os.getcwd(), 'data', 'studentVle.csv')
VLE_FPATH = os.path.join(os.getcwd(), 'data', 'vle.csv')


def ordinal_final_result(student_info):
    mod_res = student_info.sort_values(by=['id_student'])
    mod_res = mod_res[['final_result']]
    mod_res.loc[mod_res['final_result'] == 'Withdrawn', 'final_result'] = 0
    mod_res.loc[mod_res['final_result'] == 'Fail', 'final_result'] = 1
    mod_res.loc[mod_res['final_result'] == 'Pass', 'final_result'] = 2
    mod_res.loc[mod_res['final_result'] == 'Distinction', 'final_result'] = 3
    return mod_res


def assess_mark_corr(student_info):
    mod_res = ordinal_final_result(student_info)
    stud_assess = pd.read_csv(STUDENT_ASSESS_FPATH)
    assess = pd.read_csv(ASSESSMENTS_FPATH)
    stud_assess = pd.merge(stud_assess, assess, on='id_assessment')
    features = student_info[['id_student', 'code_module', 'code_presentation']]
    stud_assess_merged = pd.merge(features, stud_assess, on=[
        'id_student', 'code_module', 'code_presentation'], how='left')
    stud_assess_merged = stud_assess_merged.fillna(0)
    stud_assess_merged = stud_assess_merged.groupby(
        ['id_student', 'code_module', 'code_presentation'])['score'].sum()
    stud_assess_merged.fillna(0, inplace=True)
    res = stats.spearmanr(stud_assess_merged, mod_res)
    print('Correlation between total assessment mark for module and module' +
          ' result')
    print('Coefficient=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


def reg_data_corr(student_info):
    mod_res = ordinal_final_result(student_info)
    stud_reg = pd.read_csv(STUDENT_REG_FPATH)
    features = pd.merge(student_info, stud_reg, on=[
        'id_student', 'code_module', 'code_presentation'])
    features = features.sort_values(by=['id_student'])
    features.fillna(0, inplace=True)
    res = stats.spearmanr(features[['date_registration']], mod_res)
    print('Correlation between registration date and module result')
    print('Coefficient=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


def unreg_data_corr(student_info):
    mod_res = ordinal_final_result(student_info)
    stud_reg = pd.read_csv(STUDENT_REG_FPATH)
    features = pd.merge(student_info, stud_reg, on=[
        'id_student', 'code_module', 'code_presentation'])
    features = features.sort_values(by=['id_student'])
    features.fillna(0, inplace=True)
    res = stats.spearmanr(features[['date_unregistration']], mod_res)
    print('Correlation between de-registration date and module result')
    print('Coefficient=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


def prev_attempts_corr(student_info):
    mod_res = ordinal_final_result(student_info)
    sort_data = student_info.sort_values(by=['id_student'])
    res = stats.spearmanr(sort_data[['num_of_prev_attempts']], mod_res)
    print('Correlation between previous module attempts and module result')
    print('Coefficient=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


def stud_credits_corr(student_info):
    mod_res = ordinal_final_result(student_info)
    sort_data = student_info.sort_values(by=['id_student'])
    res = stats.spearmanr(sort_data[['studied_credits']], mod_res)
    print('Correlation between studied credits for module and module result')
    print('Coefficient=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


def course_length_corr(student_info):
    mod_res = ordinal_final_result(student_info)
    courses = pd.read_csv(COURSES_FPATH)
    features = pd.merge(student_info, courses, on=[
        'code_presentation', 'code_module'])
    features = features.sort_values(by=['id_student'])
    res = stats.spearmanr(
        features[['module_presentation_length']], mod_res)
    print('Correlation between course length and module result')
    print('Coefficient=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


def gender_association(student_info):
    freq = student_info[['gender', 'final_result']]
    freq = freq.groupby(['gender', 'final_result']).size().unstack()
    res = stats.chi2_contingency(freq)
    print('Association between gender and module result')
    print('Test stat=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


def imd_association(student_info):
    freq = student_info[['imd_band', 'final_result']]
    freq = freq.groupby(['imd_band', 'final_result']).size().unstack()
    res = stats.chi2_contingency(freq)
    print('Association between imd and module result')
    print('Test stat=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


def higher_ed_association(student_info):
    freq = student_info[['highest_education', 'final_result']]
    freq = freq.groupby(['highest_education', 'final_result']).size().unstack()
    res = stats.chi2_contingency(freq)
    print('Association between highest education level and module result')
    print('Test stat=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


def region_association(student_info):
    freq = student_info[['region', 'final_result']]
    freq = student_info.groupby(['region', 'final_result']).size().unstack()
    res = stats.chi2_contingency(freq)
    print('Association between student region and module result')
    print('Test stat=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


def disability_association(student_info):
    freq = student_info[['disability', 'final_result']]
    freq = student_info.groupby(
        ['disability', 'final_result']).size().unstack()
    res = stats.chi2_contingency(freq)
    print('Association between disability and module result')
    print('Test stat=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


def age_band_association(student_info):
    freq = student_info[['age_band', 'final_result']]
    freq = student_info.groupby(
        ['age_band', 'final_result']).size().unstack()
    res = stats.chi2_contingency(freq)
    print('Association between age and module result')
    print('Test stat=', res[0], '\np_value=', res[1])
    print('################################################################' +
          '########')


def join_tables(student_info):
    stud_reg = pd.read_csv(STUDENT_REG_FPATH)
    courses = pd.read_csv(COURSES_FPATH)
    merged = pd.merge(student_info, courses, on=[
        'code_module', 'code_presentation'])
    merged = pd.merge(merged, stud_reg, on=[
        'id_student', 'code_presentation', 'code_module'])
    return merged


def transform_for_model(data):
    features = data.drop('final_result', axis=1)
    labels = list(data['final_result'])
    return (features, labels)


def create_data_pipe(features):
    cat_features = features[['gender', 'highest_education',
                             'age_band', 'region', 'disability']]
    num_features = features[['num_of_prev_attempts', 'studied_credits',
                             'module_presentation_length',
                             'date_registration']]
    imd_features = features[['imd_band']]
    unreg_feature = features[['date_unregistration']]
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    unreg_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=999)),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('enc', OneHotEncoder())
    ])
    imd_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('enc', OrdinalEncoder())
    ])
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


def train_forest(train_set):
    joined = join_tables(train_set)
    features, labels = transform_for_model(joined)
    data_pipe = create_data_pipe(features)
    for_class = RandomForestClassifier()
    params = [
        {'n_estimators': [50], 'max_features':[5]},
    ]
    g_search = GridSearchCV(for_class, params, cv=10, scoring=[
                            'precision_micro', 'recall_micro'],
                            return_train_score=True, refit='precision_micro')
    forest_pipe = Pipeline([
        ('data_pipe', data_pipe),
        ('grid', g_search)
    ])
    forest_pipe.fit(features, labels)
    print(forest_pipe['grid'].best_params_)
    results = forest_pipe['grid'].cv_results_
    res_tuple = zip(results['mean_test_precision_micro'],
                    results['mean_test_recall_micro'], results['params'])
    for prec, rec, params in res_tuple:
        print('Prec=', prec, 'Recall=', rec, 'Params=', params)
    save_model(forest_pipe, 'forest_pipe.pkl')
    return forest_pipe


def train_svc(train_set):
    joined = join_tables(train_set)
    features, labels = transform_for_model(joined)
    data_pipe = create_data_pipe(features)
    svc = LinearSVC()
    params = [
        {'C': [0.1], 'dual':[False]},
    ]
    g_search = GridSearchCV(svc, params, cv=10, scoring=[
                            'precision_micro', 'recall_micro'],
                            return_train_score=True, refit='precision_micro')
    svc_pipe = Pipeline([
        ('data_pipe', data_pipe),
        ('grid', g_search)
    ])
    svc_pipe.fit(features, labels)
    print(svc_pipe['grid'].best_params_)
    results = svc_pipe['grid'].cv_results_
    res_tuple = zip(results['mean_test_precision_micro'],
                    results['mean_test_recall_micro'], results['params'])
    for prec, rec, params in res_tuple:
        print('Prec=', prec, 'Recall=', rec, 'Params=', params)
    save_model(svc_pipe, 'svc_pipe.pkl')
    return svc_pipe


def test_model(model, test_data, name):
    joined = join_tables(test_data)
    features, labels = transform_for_model(joined)
    preds = model.predict(features)
    precision = precision_score(labels, preds, average='micro')
    accuracy = accuracy_score(labels, preds)
    conf = confusion_matrix(labels, preds)
    conf_plot = plot_confusion_matrix(model, features, labels, normalize=None,
                          values_format='.9g', cmap=plt.cm.Blues)
    conf_plot.ax_.set_title(name)
    print('Precision=', precision, '\nAccuracy=', accuracy)
    print('Confusion Matrix:')
    print(conf)


def save_model(model, name):
    joblib.dump(model, name)


def load_model(name):
    model = joblib.load(name)
    return model


def main():
    student_info = pd.read_csv(STUDENT_INFO_FPATH)
    train_set, test_set = train_test_split(
        student_info, test_size=0.2, random_state=42,
        stratify=student_info['highest_education'])
    # process_data(train_set)
    rand_forest = train_forest(train_set)
    test_model(rand_forest, test_set, 'Forest')
    svc = train_svc(train_set)
    test_model(svc, test_set, 'SVC')
    plt.show()


main()
