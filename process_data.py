import os
import numpy as np
import pandas as pd
from scipy import stats

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
    mod_res = mod_res[mod_res['final_result'] != 'Withdrawn'][['final_result']]
    mod_res.loc[mod_res['final_result'] == 'Fail', 'final_result'] = 1
    mod_res.loc[mod_res['final_result'] == 'Pass', 'final_result'] = 2
    mod_res.loc[mod_res['final_result'] == 'Distinction', 'final_result'] = 3
    return mod_res


def assess_mark_corr(student_info):
    mod_res = ordinal_final_result(student_info)
    stud_assess = pd.read_csv(STUDENT_ASSESS_FPATH)
    assess = pd.read_csv(ASSESSMENTS_FPATH)
    stud_assess = pd.merge(stud_assess, assess, on='id_assessment')
    non_withdraw = student_info[student_info['final_result'] != 'Withdrawn'][[
        'id_student', 'code_module', 'code_presentation']]
    stud_assess_merged = pd.merge(non_withdraw, stud_assess, on=[
        'id_student', 'code_module', 'code_presentation'], how='left')
    stud_assess_merged = stud_assess_merged.fillna(0)
    stud_assess_merged = stud_assess_merged.groupby(
        ['id_student', 'code_module', 'code_presentation'])['score'].sum()
    print(stats.spearmanr(stud_assess_merged, mod_res))


def prev_attempts_corr(student_info):
    mod_res = ordinal_final_result(student_info)
    non_withdraw = student_info[student_info['final_result'] != 'Withdrawn']
    non_withdraw = non_withdraw.sort_values(by=['id_student'])
    print(stats.spearmanr(non_withdraw[['num_of_prev_attempts']], mod_res))


def stud_credits_corr(student_info):
    mod_res = ordinal_final_result(student_info)
    non_withdraw = student_info[student_info['final_result'] != 'Withdrawn']
    non_withdraw = non_withdraw.sort_values(by=['id_student'])
    print(stats.spearmanr(non_withdraw[['studied_credits']], mod_res))


def course_length_corr(student_info):
    mod_res = ordinal_final_result(student_info)
    courses = pd.read_csv(COURSES_FPATH)
    non_withdraw = student_info[student_info['final_result'] != 'Withdrawn']
    non_withdraw = pd.merge(non_withdraw, courses, on=[
                            'code_presentation', 'code_module'])
    non_withdraw = non_withdraw.sort_values(by=['id_student'])
    print(stats.spearmanr(
        non_withdraw[['module_presentation_length']], mod_res))


def gender_association(student_info):
    freq = student_info[['gender', 'final_result']]
    freq = freq.groupby(['gender', 'final_result']).size().unstack()
    print(stats.chi2_contingency(freq))


def imd_association(student_info):
    freq = student_info[['imd_band', 'final_result']]
    freq = freq.groupby(['imd_band', 'final_result']).size().unstack()
    print(stats.chi2_contingency(freq))


def higher_ed_association(student_info):
    freq = student_info[['highest_education', 'final_result']]
    freq = freq.groupby(['highest_education', 'final_result']).size().unstack()
    print(stats.chi2_contingency(freq))


def region_association(student_info):
    freq = student_info[['region', 'final_result']]
    freq = student_info.groupby(['region', 'final_result']).size().unstack()
    print(stats.chi2_contingency(freq))


def disability_association(student_info):
    freq = student_info[['disability', 'final_result']]
    freq = student_info.groupby(
        ['disability', 'final_result']).size().unstack()
    print(stats.chi2_contingency(freq))


def process_data():
    student_info = pd.read_csv(STUDENT_INFO_FPATH)
    assess_mark_corr(student_info)
    prev_attempts_corr(student_info)
    stud_credits_corr(student_info)
    course_length_corr(student_info)
    gender_association(student_info)
    imd_association(student_info)
    higher_ed_association(student_info)
    region_association(student_info)
    disability_association(student_info)


process_data()
