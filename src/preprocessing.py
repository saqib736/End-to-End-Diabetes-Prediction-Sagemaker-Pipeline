import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

def preprocess(input_path, train_output_path, test_output_path, test_size=0.2, random_state=42):
    # Load dataset
    df = pd.read_csv(input_path)

    # Map categorical and binary variables
    binary_mapping = {'Yes': 1, 'No': 0}
    
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['class'] = df['class'].map({'Positive': 1, 'Negative': 0})
    
    binary_cols = df.columns.drop(['Age', 'Gender', 'class'])
    for col in binary_cols:
        df[col] = df[col].map(binary_mapping)

    # Feature scaling for 'Age'
    df['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()

    # Split data into training and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Save preprocessed training and test data
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/raw_data.csv')
    parser.add_argument('--train_output_path', type=str, default='data/train_data.csv')
    parser.add_argument('--test_output_path', type=str, default='data/test_data.csv')
    args = parser.parse_args()
    preprocess(args.input_path, args.train_output_path, args.test_output_path)
