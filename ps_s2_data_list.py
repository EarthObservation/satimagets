# coding: utf-8

# List PlanetScope and Sentinel-2 data and write to Excel
#
# Krištof Oštir
# UL FGG
# (c) 2020

# %%
# Imports
import pandas as pd
import os

# %%
# Folders
ps_root = 'x:\\PS\PS_5m'
s2_root = 'x:\\S2_10m'

# %%
# Image database
ps_list_fn ='PS_data.xlsx'
s2_list_fn ='S2_data.xlsx'

#%%
# PS files to DF
ps_df = pd.DataFrame()
ps_columns = ['PS_AOI', 'PS_Date', 'PS_filename', 'PS_Sat', 'PS_Path']
for root, dirs, files in os.walk(ps_root):
    file_list = []
    for filename in files:
        if filename.endswith('_analytic.tif'):
            ps_aoi = filename[0:2]
            ps_date = filename[3:11]
            ps_id = filename[-17:-13]
            file_list.append([ps_aoi, ps_date, filename, ps_id, os.path.join(root, filename)])
    ps_folder_df = pd.DataFrame(file_list, columns=ps_columns)
    ps_df = ps_df.append(ps_folder_df)
# Create date column, all others are string
ps_df = ps_df.astype(str)
ps_df['PS_Date'] = pd.to_datetime(ps_df['PS_Date']).dt.date

#%%
# S2 files to DF
s2_df = pd.DataFrame()
s2_columns = ['S2_Date', 'S2_filename', 'S2_Sat', 'S2_Orbit', 'S2_path']
for root, dirs, files in os.walk(s2_root):
    file_list = []
    for filename in files:
        if filename.endswith('10m__ms_d96tm.tif'):
            s2_date = filename[0:8]
            s2_id = filename[16:19]
            s2_orb = filename[43:47]
            file_list.append([s2_date, filename, s2_id, s2_orb, os.path.join(root, filename)])
    s2_folder_df = pd.DataFrame(file_list, columns=s2_columns)
    s2_df = s2_df.append(s2_folder_df)
# Create date column, all others are string
s2_df = s2_df.astype(str)
s2_df['S2_Date'] = pd.to_datetime(s2_df['S2_Date']).dt.date

#%%
# Save lists to file
ps_df.to_excel(ps_list_fn, index=False)
s2_df.to_excel(s2_list_fn,index=False)
