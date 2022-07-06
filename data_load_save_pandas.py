import pandas as pd
import os


def save_csv(path, file_name, df):
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, file_name))


def load_csv(path, file_name):
    file = os.path.join(path, file_name)
    if os.path.isfile(file):
        df = pd.read_csv(file)
        return df
    else:
        raise Exception('{} file does not exist'.format(file))



