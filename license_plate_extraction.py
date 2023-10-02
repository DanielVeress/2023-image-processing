import os
import pandas as pd
from utils.constants import PROCESSED_DATABASE_FILE


df = pd.read_csv(PROCESSED_DATABASE_FILE)


print(df.shape)