

import streamlit as st
import pandas as pd
from cgan_model2 import CGAN  # Make sure this matches the name of the file where your CGAN class is defined
from io import StringIO
import base64

def main():
    st.title('IoT Network Traffic Synthetic Data Generator')

    # Initialize session state variables if they are not already present
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'synthetic_data' not in st.session_state:
        st.session_state.synthetic_data = None

    # File upload section
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.session_state.df = pd.read_csv(stringio)
        # Strip trailing and leading spaces from column names
        st.session_state.df.columns = st.session_state.df.columns.str.strip()
        st.write("First 5 rows of your dataset:")
        st.dataframe(st.session_state.df.head())

    # Check if the dataframe is loaded
    if st.session_state.df is not None:
        # User inputs for data configuration
        st.sidebar.header('Data Configuration')
        class_column = st.sidebar.selectbox("Select the class label column", st.session_state.df.columns)
        categorical_cols = st.sidebar.multiselect("Select categorical features", st.session_state.df.columns)
        numerical_cols = st.sidebar.multiselect("Select numerical features", st.session_state.df.columns)

        # User inputs for integer columns without decimals
        int_columns = st.sidebar.multiselect("Select numerical features to preserve as integers without decimals", numerical_cols)

        # Model configuration
        st.sidebar.header('Model Configuration')
        epochs = st.sidebar.number_input("Number of epochs", min_value=1, value=50, step=1)
        num_samples = st.sidebar.number_input("Number of synthetic samples to generate", min_value=1, value=1000, step=1)

        # Generate synthetic data
        if st.sidebar.button("Generate Synthetic Data"):
            if not class_column or not categorical_cols or not numerical_cols:
                st.sidebar.error("Please select class label and features correctly!")
            else:
                cgan = CGAN(categorical_cols, numerical_cols, class_column, int_columns=int_columns, latent_dim=100)
                df_preprocessed = cgan.preprocess_data(st.session_state.df)
                cgan.train(df_preprocessed, num_samples, batch_size=32, epochs=epochs, learning_rate=0.0002)
                st.session_state.synthetic_data = cgan.generate_samples(num_samples)

                st.write("Synthetic Data Generated:")
                st.dataframe(st.session_state.synthetic_data.head())

                # Download link for generated data
                csv = st.session_state.synthetic_data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="synthetic_data.csv">Download csv file</a>'
                st.markdown(href, unsafe_allow_html=True)

                # Placeholder for evaluation function
                # if st.button("Start Evaluation"):
                #     progress_bar = st.progress(0)
                #     progress_text = st.empty()
                #     accuracy, report_df = evaluate(st.session_state.df, st.session_state.synthetic_data, categorical_cols, numerical_cols, class_column, epochs, progress_bar=progress_report, progress_text=progress_text)
                #     st.success('Evaluation completed!')

if __name__ == "__main__":
    main()
