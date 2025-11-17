import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from google import genai
import os 

API='AIzaSyBNsdENpvh4cTNdgGBQ_zKk09UlYzHsbvo'

base_path = os.getcwd()

df = pd.read_excel(os.path.join(base_path, 'data', 'biased_prompt_1.xlsx'))







