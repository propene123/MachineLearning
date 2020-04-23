import os
import numpy as np
import pandas as pd

ASSESSMENTS_FPATH = os.path.join(os.getcwd(), 'data', 'assessments.csv')
COURSES_FPATH = os.path.join(os.getcwd(), 'data', 'courses.csv')
STUDENT_ASSESS_FPATH = os.path.join(
    os.getcwd(), 'data', 'studentAssessment.csv')
STUDENT_INFO_FPATH = os.path.join(os.getcwd(), 'data', 'studentInfo.csv')
STUDENT_REG_FPATH = os.path.join(
    os.getcwd(), 'data', 'studentRegistration.csv')
STUDENT_VLE_FPATH = os.path.join(os.getcwd(), 'data', 'studentVle.csv')
VLE_FPATH = os.path.join(os.getcwd(), 'data', 'vle.csv')


df = pd.read_csv(STUDENT_INFO_FPATH)
print(df['id_student'].value_counts())
fd = pd.read_csv(STUDENT_ASSESS_FPATH)
print(fd['id_student'].value_counts())
