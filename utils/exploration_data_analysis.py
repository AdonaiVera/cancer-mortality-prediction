import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_dataset():
    """
    Performs an exploratory data analysis (EDA) on the cancer dataset and answers key questions.
    
    Returns:
    None (prints information about the dataset)
    """
    
    # Step 1: Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('data/cancer_reg.csv', encoding='ISO-8859-1')
    
    # 1) How many data samples are included in the dataset?
    num_samples = df.shape[0]
    print(f"1) Number of data samples: {num_samples}")
    
    # 2) Which problem will this dataset try to address?
    print("2) The dataset addresses a supervised learning problem where the goal is to predict cancer mortality rates, labeled as 'TARGET_deathRate'.")
    print("   - This is a **regression** problem, where the label ('TARGET_deathRate') represents a continuous value, specifically the mortality rate due to cancer in different regions.")
    print("   - The dataset contains various features (such as demographic and healthcare-related data) that can be used to make predictions about the mortality rate.")
    print("   - Our task is to use the provided features to build models that accurately predict the 'TARGET_deathRate'.")
    
    
    # 3) What is the minimum value and the maximum value in the dataset?
    print("3) Minimum and maximum values in the dataset:")
    # Get the minimum and maximum values for each column
    min_values = df.min(numeric_only=True)
    max_values = df.max(numeric_only=True)
    
    # Print the minimum and maximum values for each feature
    for column in min_values.index:
        print(f"   - {column}: Min = {min_values[column]}, Max = {max_values[column]}")
    
    # 4) How many features in each data sample?
    num_features = df.shape[1] - 1  # Exclude the target column
    print(f"4) Number of features per data sample (excluding target): {num_features}")
    
    # 5) Does the dataset have any missing information? E.g., missing features.
    missing_info = df.isnull().sum()
    missing_features = missing_info[missing_info > 0]
    print(f"5) Missing information in the dataset:\n{missing_features if not missing_features.empty else 'No missing data'}")
    
    # 6) What is the label of this dataset?
    print("6) The label of this dataset is 'TARGET_deathRate', which represents cancer mortality rates.")
    
    # 7) How many percent of data will you use for training, validation, and testing?
    train_percent = 0.7
    val_percent = 0.15
    test_percent = 0.15
    print(f"7) We will use {train_percent*100}% for training, {val_percent*100}% for validation, and {test_percent*100}% for testing.")
    
    # 8) What kind of data pre-processing will you use for your training dataset?
    print("8) Planned data pre-processing:")
    print("   - Drop non-numeric columns.")
    print("   - Handle missing values by filling them with the mean.")
    print("   - Normalize the features using StandardScaler.")
    print("   - Split the dataset into training, validation, and test sets.")

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

    # Save the plot as a PNG file
    plt.title('Correlation Matrix Heatmap')
    plt.savefig('figures/correlation_matrix_heatmap.png', dpi=300)
    plt.close()  # Close the plot to avoid displaying in the notebook

    print("Correlation matrix heatmap saved as 'figures/correlation_matrix_heatmap.png'")


if __name__ == "__main__":
    explore_dataset()
