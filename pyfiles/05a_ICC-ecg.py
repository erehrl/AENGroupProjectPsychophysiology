#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Load your data (replace with your actual path)
results_folder = '/Users/erwin/Documents/ProjectPsychophysiologyData/results/'
filename = results_folder + 'ecg_results.csv'
results = pd.read_csv(filename)

# Ensure you have columns for participant, condition, and your measure (e.g., ECG_Rate_Mean)
# Let's assume 'Condition' is your independent variable and 'ECG_Rate_Mean' is the dependent variable.

# Fit the model: 
# We want to calculate ICC for 'ECG_Rate_Mean' for different 'Condition' and 'Participant' 
# Here, 'Condition' is fixed, and 'Participant' is random (subject effect).
model = ols('ECG_Rate_Mean ~ C(Condition) + C(Participant)', data=results).fit()

# Perform ANOVA to get variance components
anova_table = anova_lm(model, typ=2)
print(anova_table)

# ICC is calculated as variance between participants / (variance between participants + variance within participants)
# Get variance components from ANOVA table
between_variance = anova_table['sum_sq']['C(Participant)']
within_variance = anova_table['sum_sq']['Residual']

# Calculate ICC
icc = between_variance / (between_variance + within_variance)

# Convert ICC to percentage
icc_percentage = icc * 100

# Print the ICC and its interpretation
print(f"Intraclass Correlation Coefficient (ICC): {icc:.4f}")
print(f"ICC as a percentage: {icc_percentage:.2f}%")

# Interpretation sentence
interpretation = (f"The ICC indicates that {icc_percentage:.2f}% of the variability "
                  "in mean ECG rates is due to differences between participants, rather than differences between conditions (within participants).")

print(interpretation)


# In[ ]:




