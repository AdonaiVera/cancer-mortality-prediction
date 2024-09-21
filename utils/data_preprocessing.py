import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import skew
import numpy as np

def load_and_preprocess_data_base(apply_pca=False, n_components=0.95):
    """
    Loads the cancer mortality dataset, handles missing values, splits the data into
    training, validation, and test sets, and normalizes the features. Optionally applies PCA.
    
    Parameters:
    apply_pca (bool): If True, applies PCA for dimensionality reduction.
    n_components (float or int): Number of components to keep in PCA. If float, it represents the amount of variance to keep.
    
    Returns:
    X_train, X_val, X_test, y_train, y_val, y_test: Arrays containing the split and preprocessed data.
    """
    
    # Step 1: Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('data/cancer_reg.csv', encoding='ISO-8859-1')
    
    # Step 2: Handle non-numeric columns
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    print(f"Non-numeric columns identified and dropped: {non_numeric_cols}")
    df = df.drop(columns=non_numeric_cols)
    
    # Step 3: Handle missing values by filling with the mean of each column
    print("Filling missing values with the mean of their respective columns...")
    df.fillna(df.mean(), inplace=True)
    
    
    # Step 4: Normalize the features using StandardScaler
    print("Normalizing the features using StandardScaler...")
    ss = StandardScaler()
    df_new = ss.fit_transform(df)
    df_optimized = pd.DataFrame(columns=df.columns, data=df_new)

    # Step 5: Separate features (X) and target (y)
    X_scaled = df_optimized.drop(columns=['TARGET_deathRate'])
    y = df_optimized['TARGET_deathRate']
    
    # Step 6: Apply PCA if apply_pca is True
    if apply_pca:
        print(f"Applying PCA with n_components={n_components}...")
        pca = PCA(n_components=n_components)
        X_scaled = pca.fit_transform(X_scaled)
        print(f"Explained variance ratio by PCA: {pca.explained_variance_ratio_.sum()}")

    # Step 7: Split the data into training, validation, and test sets
    print("Splitting the dataset into training, validation, and test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)  # 70% training
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% validation, 15% test
    
    print(f"Data split completed: {len(X_train)} training samples, {len(X_val)} validation samples, {len(X_test)} test samples.")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_and_preprocess_data(apply_pca=False, n_components=0.95):
    """
    Loads the cancer mortality dataset, handles missing values, splits the data into
    training, validation, and test sets, and normalizes the features. Optionally applies PCA.
    
    Parameters:
    apply_pca (bool): If True, applies PCA for dimensionality reduction.
    n_components (float or int): Number of components to keep in PCA. If float, it represents the amount of variance to keep.
    
    Returns:
    X_train, X_val, X_test, y_train, y_val, y_test: Arrays containing the split and preprocessed data.
    """
    
    # Step 1: Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('data/cancer_reg.csv', encoding='ISO-8859-1')

    # Removing the 'Geography' column
    df = df.drop(columns=['Geography', 'PctSomeCol18_24', 'binnedInc'])

    # Select numeric columns only (exclude boolean columns if any)
    numeric_feats = df.select_dtypes(include=[np.number]).columns

    # Compute skewness and filter features with skewness > 0.75
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75].index

    # Apply log1p transformation to skewed features to reduce skewness
    df[skewed_feats] = np.log1p(df[skewed_feats])

    df['PctEmployed16_Over'].fillna(df['PctEmployed16_Over'].median(), inplace=True)
    df['PctPrivateCoverageAlone'].fillna(df['PctPrivateCoverageAlone'].median(), inplace=True)

    print("Columns")
    print(df.columns)

    # Step 4: Normalize the features using StandardScaler
    ss = MinMaxScaler()
    df_new = ss.fit_transform(df)
    df_optimized = pd.DataFrame(columns=df.columns, data=df_new)

    print(df_optimized)

    # Step 4: Separate features (X) and target (y)
    X_scaled = df_optimized.drop(columns=['TARGET_deathRate'])
    y = df_optimized['TARGET_deathRate']
        
    # Step 5: Apply PCA if apply_pca is True
    if apply_pca:
        print(f"Applying PCA with n_components={n_components}...")
        pca = PCA(n_components=n_components)
        X_scaled = pca.fit_transform(X_scaled)
        print(f"Explained variance ratio by PCA: {pca.explained_variance_ratio_.sum()}")

    # Step 6: Split the data into training, validation, and test sets
    print("Splitting the dataset into training, validation, and test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  # 70% training
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% validation, 15% test
    
    print(f"Data split completed: {len(X_train)} training samples, {len(X_val)} validation samples, {len(X_test)} test samples.")
    
    return X_train, X_val, X_test, y_train, y_val, y_test