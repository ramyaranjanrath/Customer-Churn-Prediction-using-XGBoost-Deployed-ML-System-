import pandas as pd


def map_binary_series(s: pd.Series) -> pd.Series:
    """
    For binary columns map them to 0/1
    """
    #Get unique values from the series and remove NaN values
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    #Yes/No mapping
    if valset == {'Yes','No'}:
        return s.map({'Yes':1,'No':0}).astype("Int64")
    
    #Gender mapping
    if valset == {'Male', 'Female'}:
        return s.map({'Male':1,'Female':0}).astype("Int64")
    
    #Alphabetical ordering for other binary categories
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]:0, sorted_vals[1]:1}
        return s.astype(str).map(mapping).astype("Int64")
    
    return s
    
def build_features(df:pd.DataFrame, target_column:str = "Churn") -> pd.DataFrame:
    """
    explanation
    """

    df = df.copy()
    print(f"starting feature engineering on {df.shape[1]} columns")

    #Step1: Find categorical and numerical columns
    obj_cols = [c for c in df.select_dtypes(include=['object']).columns if c != target_column]
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    print(f"found {len(obj_cols)} categorical and {len(numeric_cols)} numerical columns")

    #Step2: Split categorical columns according to the number of unique values
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    print(f"binary features: {len(binary_cols)} and multi-category features: {len(multi_cols)}")

    #Step3: Apply binary encoding
    for c in binary_cols:
        df[c] =  map_binary_series(df[c].astype(str))
    
    #Step4: One Hot Encoding for multi-category features
    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    
    #Step5: Convert boolean to int
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    #Step6: Convert nullable integers (Int64) to standard integers
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c]=df[c].fillna(0).astype(int)

    print(f"Feature engineering complete final features: {df.shape[1]}")
    return df



