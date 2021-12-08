# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 19:26:10 2021

@author: mabul
"""

import pandas as pd

df = pd.read_json('C:/Users/mabul/OneDrive/Documents/Gio/UMN MABA\Machine Learning/meta_Video_Games.json/meta_Video_Games.json', lines=True)

df.to_csv('C:/Users/mabul/OneDrive/Documents/Gio/UMN MABA\Machine Learning/meta_Video_Games.json/Video-games.csv')
