import pandas as pd

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    - Trim column names.
    - Drop customerID column.
    - Change TotalCharges column to numeric datatype.
    - Map target_column to 0/1.
    - Fill missing values in numerical columns with 0.
    
    """

    #Remove spaces from column headers
    df.columns =  df.columns.str.strip()  # Remove leading/trailing whitespace

    #Drop the customerID column
    df = df.drop('customerID', axis=1)

    #Change the TotalCharges column to numeric.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    #Change the target column to 0/1
    df[target_col] = df[target_col].str.strip().map(
        {
            'Yes':1,
            'No': 0,
        }
    )
    #For numerical columns fill missing values with 0
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df