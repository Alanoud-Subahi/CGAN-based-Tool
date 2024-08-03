import os
import streamlit as st
import subprocess

# Define the directory where your scripts are located
SCRIPTS_DIR = os.path.dirname(__file__)

def main():
    st.title('App Landing Page')
    st.write('Welcome to My App!')
    st.write('This app includes various functionalities.')
    st.write('Choose from the options below:')
    
    if st.button('IoT Network Traffic Synthetic Data Generator'):
        st.write('Redirecting to IoT Network Traffic Synthetic Data Generator...')
        run_script('main2.py')

    if st.button('Exploratory Data Analysis'):
        st.write('Redirecting to Exploratory Data Analysis...')
        run_script('main3.py')

def run_script(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    command = f'streamlit run {script_path}'
    subprocess.Popen(command, shell=True)

if __name__ == '__main__':
    main()
