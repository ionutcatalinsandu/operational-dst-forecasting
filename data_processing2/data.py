import pandas as pd

from data_processing2.constants import *
from data_processing.constants import COLUMNS_WITH_THRESHOLD
from data_processing.data import replace_outliers_with_mean, replace_outliers_with_nan, create_time_column


def read_data(format_path: str, lst_path: str):
    column_names = pd.read_fwf(format_path)
    column_names = column_names["FORMAT OF THE SUBSETTED FILE"].values[1:]
    
    # Extract column name (between first and last space)
    parsed_names = []
    for col in column_names:
        col = col.strip()
        # Remove number prefix (before first space)
        col = col[col.find(" ")+1:]
        # Remove format suffix (after last space)
        col = col[:col.rfind(" ")].strip()
        parsed_names.append(col)
    
    column_names = parsed_names

    df = pd.read_csv(lst_path, header=None, sep=r'\s+', engine='python')
    column_mapper = {i: name for i, name in enumerate(column_names)}
    df.rename(columns=column_mapper, inplace=True)

    good_columns = [name for name in column_names if name not in BAD_COLUMNS]
    x = df[good_columns].copy()
    x = create_time_column(x, FULL_DATE_COLUMN)
    
    return {
        "x": x,
        "column_names": column_names,
        "good_columns": good_columns,
    }

def process_dataset(dataset: pd.DataFrame, replace_outliers=True, print_max_values=False):

    dataset = dataset.copy()
    if replace_outliers:
        print("Replacing outliers...")
        for column, max_value in COLUMNS_WITH_THRESHOLD:
            dataset = replace_outliers_with_mean(dataset, column, max_value, print_count=False)
        print("Done replacing outliers with mean.")
    else:
        print("Replacing outliers with nan...")
        for column, max_value in COLUMNS_WITH_THRESHOLD:
            dataset = replace_outliers_with_nan(dataset, column, max_value)
        print("Done replacing outliers with nan.")

    # check if there are any strange outliers
    if print_max_values:
        for column in dataset.columns:
            print(column, " - Max value: ", dataset[column].max())
        # done checking

    # dataset = create_time_column(dataset, FULL_DATE_COLUMN)

    # return correct columns
    dataset = dataset[CANDIDATE_COLUMNS].copy()

    # dataset = convert_angular_features(dataset, ANGULAR_TRIGONOMETRIC, "degrees", True)
    # dataset.dropna(inplace=True)
    
    return dataset