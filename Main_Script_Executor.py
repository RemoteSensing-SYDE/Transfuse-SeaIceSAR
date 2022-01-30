"""
No@
"""
import os

Schedule = []

Schedule.append("python train_isic.py --train_path Scene58/ \
                                ")

Schedule.append("python test_isic.py --test_path Scene56/ \
                                ")

for i in range(len(Schedule)):
    os.system(Schedule[i])