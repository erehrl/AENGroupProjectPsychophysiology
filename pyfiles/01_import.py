#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from opensignalsreader import OpenSignalsReader
import os


# In[3]:


# Parameters
sourceDataFolder = '/Users/erwin/Documents/ProjectPsychophysiologyData/source-data'
rawDataFolder = '/Users/erwin/Documents/ProjectPsychophysiologyData/raw-data/'
participants = ['sub-1', 'sub-2', 'sub-3']
tasks = ['baseline', 'spiderhand', 'spidervideo']

# Ensure output directory exists
if not os.path.exists(rawDataFolder):
    os.makedirs(rawDataFolder)

# Define the EDA transfer function
def eda(samples, resolution=10):
    """
    Converts raw EDA values into original physical unit (µS).
    EDA(µS) = ((((ADC/2^n) * VCC) - 0.574) / 0.132)
    """
    if samples is None:
        raise ValueError("No input samples provided for EDA conversion.")

    # Compute EDA in µS
    eda_samples = np.asarray([(((float(s) / (2 ** resolution)) * 3.3) - 0.574) / 0.132 for s in samples])
    
    # Optionally, check and filter ranges (e.g., -4.4µS to 21µS) if needed
    eda_samples = np.clip(eda_samples, -4.4, 21.0)
    
    return eda_samples

for pi in participants:
    for ti in tasks:
        # Assemble file name 
        filename = sourceDataFolder + '/' + pi + '_' + ti + '.txt'
        
        # Convert all other files 
        acq = OpenSignalsReader(filename, show=True)
        
        # Access the processed ECG / EDA signal
        ecg_signal = acq.signal(1)
        print(f"First 5 elements of ECG signal for {pi}, {ti}: {ecg_signal[:5]}")
        
        eda_signal_raw = acq.signal(2)
        print(f"First 5 elements of raw EDA signal for {pi}, {ti}: {eda_signal_raw[:5]}")
        
        # Apply the EDA transfer function
        eda_signal_transformed = eda(eda_signal_raw)
        print(f"First 5 elements of transformed EDA signal for {pi}, {ti}: {eda_signal_transformed[:5]}")
        
        # Save data in a .csv format 
        output_filename_ecg = rawDataFolder + '/' + pi + '_' + ti + '_ecg.csv'
        output_filename_eda = rawDataFolder + '/' + pi + '_' + ti + '_eda.csv'
    
        # Save the NumPy array to a CSV file
        np.savetxt(output_filename_ecg, ecg_signal, delimiter=',', header='ECG')
        np.savetxt(output_filename_eda, eda_signal_transformed, delimiter=',', header='EDA')
    
        print(f"Wrote files {output_filename_ecg} and {output_filename_eda}")


# In[ ]:




