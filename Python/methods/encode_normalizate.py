from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from pandas import DataFrame

def coding_normalization(dataset):
    """
    Encodes categorical features and normalizes all features in a dataset.
    
    Args:
        dataset (pd.DataFrame): Input dataset to process.
    
    Returns:
        pd.DataFrame: Processed dataset with encoded and normalized values.
    """
    dataset = dataset.fillna(0)

    # Detect categorical features
    def detect_categorical(df):
        # Non-numeric columns
        non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
        # Include numeric columns with low cardinality
        low_cardinality_cols = [
            col for col in df.columns if df[col].nunique() < 10 and df[col].dtype in ['int64', 'float64']
        ]
        return list(set(non_numeric_cols + low_cardinality_cols))

    categorical_features = detect_categorical(dataset)
    
    # Convert categorical columns to strings to ensure uniformity
    for col in categorical_features:
        dataset[col] = dataset[col].astype(str)
    
    # Apply OrdinalEncoder to categorical features
    encoder = OrdinalEncoder()
    if categorical_features:
        dataset[categorical_features] = encoder.fit_transform(dataset[categorical_features])
    
    # Apply MinMaxScaler to normalize all features
    scaler = MinMaxScaler()
    dataset_scaled = DataFrame(
        scaler.fit_transform(dataset),
        columns=dataset.columns
    )
    
    return dataset_scaled

