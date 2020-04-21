import os
import numpy as np
import pandas as pd

ASSESSMENTS_FPATH = os.path.join(os.getcwd(), 'data', 'assessments.csv')
COURSES_FPATH = os.path.join(os.getcwd(), 'data', 'courses.csv')
STUDENT_ASSESS_FPATH = os.path.join(
    os.getcwd(), 'data', 'studentAssessment.csv')
STUDENT_INFO_FPATH = os.path.join(os.getcwd(), 'data', 'studentAssessment.csv')
STUDENT_REG_FPATH = os.path.join(
    os.getcwd(), 'data', 'studentRegistration.csv')
STUDENT_VLE_FPATH = os.path.join(os.getcwd(), 'data', 'studentVle.csv')
VLE_FPATH = os.path.join(os.getcwd(), 'data', 'vle.csv')

def merge_data():
    assessments = pd.read_csv(ASSESSMENTS_FPATH)
    courses = pd.read_csv(COURSES_FPATH)
    student_assess = pd.read_csv(STUDENT_ASSESS_FPATH)
    student_info = pd.read_csv(STUDENT_INFO_FPATH)
    student_reg = pd.read_csv(STUDENT_REG_FPATH)
    student_vle = pd.read_csv(STUDENT_VLE_FPATH)
    vle = pd.read_csv(VLE_FPATH)
