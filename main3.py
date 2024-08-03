import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Disable the warning
st.set_option('deprecation.showPyplotGlobalUse', False)

def perform_eda(generated_data):
    # Display basic statistics
    st.write("Dataset Description:")
    st.write(generated_data.describe())

    # Visualize distributions of numerical features
    st.write("Distributions of Numerical Features:")
    num_features = [col for col in generated_data.columns if generated_data[col].dtype != 'object']
    num_plots = len(num_features)
    num_rows = (num_plots // 3) + (1 if num_plots % 3 != 0 else 0)
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    for i, column in enumerate(num_features):
        row = i // 3
        col = i % 3
        sns.histplot(generated_data[column], kde=True, ax=axes[row, col])
        axes[row, col].set_title(column)
    st.pyplot(fig)

    # Visualize categorical features
    st.write("Categorical Features:")
    cat_features = [col for col in generated_data.columns if generated_data[col].dtype == 'object']
    for column in cat_features:
        st.write(f"Counts for '{column}':")
        plt.figure(figsize=(8, 6))  # Create a new figure for each count plot
        sns.countplot(y=column, data=generated_data)
        st.pyplot()  # Pass the current figure to st.pyplot()

def feature_engineering(generated_data, class_column, selected_cat_features):
    # Encode categorical features
    for feature in selected_cat_features:
        encoder = LabelEncoder()
        generated_data[feature] = encoder.fit_transform(generated_data[feature])

    # Train a Random Forest classifier to determine feature importances
    X = generated_data.drop(columns=[class_column])
    y = generated_data[class_column]
    clf = RandomForestClassifier()
    clf.fit(X, y)

    # Get feature importances
    feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Display feature importances
    st.write("Feature Importances:")
    st.write(feature_importances[:5])  # Display the top 5 most important features

def main():
    st.title('Exploratory Data Analysis')

    # File upload
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    # Check if file is uploaded
    if uploaded_file is not None:
        # Load the uploaded data into a DataFrame
        generated_data = pd.read_csv(uploaded_file)

        # Perform EDA
        perform_eda(generated_data)

        # Select class column
        class_column = st.selectbox("Select Class Column", generated_data.columns)

        # Check if class column is selected
        if class_column:
            # Select categorical features
            selected_cat_features = st.multiselect("Select Categorical Features", 
                                                   [col for col in generated_data.columns if generated_data[col].dtype == 'object'])

            # Submit button to proceed
            if st.button("Submit"):
                # Count labels per class in the selected class column
                class_counts = generated_data[class_column].value_counts()
                st.write("Counts per Class:")
                st.write(class_counts)

                # Feature engineering
                feature_engineering(generated_data, class_column, selected_cat_features)

if __name__ == "__main__":
    main()
