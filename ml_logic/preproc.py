import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler


# clean data: rem dupes and nas
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - removing duplicates
    - removing NAs from the largest df axis
    """
    # choose to drop nas from columns or rows based off which axis is bigger
    if df.shape[0] > df.shape[1]:
        df_nonas = df.dropna(how="any", axis=0)
    else:
        df_nonas = df.dropna(how="any", axis=1)
    # remove duplicates for our final version of a clean df
    df_cleaned = df_nonas.drop_duplicates()

    print("✅ data cleaned")

    return df_cleaned


# apply standard scaling
def scale_data(df):
    scaler = StandardScaler()

    scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled)  # .iloc[:100] IS THIS EVEN NECESSARY? HMMM

    print("✅ data scaled")

    return df_scaled


# apply digit frquency transformation
def df_count_occurences(digit, df):
    def extract_digit(number):
        try:
            return str(number).split(".")[-1][digit]
        except:
            # case when there is no second digit
            return 0

    values_df = df.applymap(extract_digit)

    # count occurrences of each digit in a row
    def count_digits_row(row):
        # concat all values in the row into a single string
        row_as_str = "".join(row.astype(str))
        return Counter(row_as_str)

    # apply function to each row
    digit_occurrences_per_row = values_df.apply(count_digits_row, axis=1)

    digits = list(range(10))

    # add missing digits with 0 value
    def add_missing_digits(digit_dict):
        for digit in digits:
            if str(digit) not in digit_dict:
                digit_dict[str(digit)] = 0
        del digit_dict[str(20)]
        return digit_dict

    # apply function to each item in series
    completed_series = digit_occurrences_per_row.apply(add_missing_digits)
    df_from_series = pd.DataFrame(completed_series.tolist())  # .dropna(axis=1)

    return df_from_series
