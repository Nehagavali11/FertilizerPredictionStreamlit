🌱 Fertilizer Recommendation System
A Streamlit web application that predicts optimal fertilizer based on soil and crop parameters using a trained Machine Learning model. Users can input data, get predictions, and view explanatory visualizations.

How to Run Locally
Ensure you have the dataset and scripts:
Place fertilizer_training_dataset_3000.csv, model_training.py, and streamlit_app.py in the same directory.

Set up environment & install dependencies:
Create a requirements.txt file:

streamlit
pandas
scikit-learn
matplotlib
seaborn
joblib

Then, install:

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate.bat  # Windows CMD
# .venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt

Train the Model:

python model_training.py

This creates the models/ directory.

Run the App:

streamlit run streamlit_app.py
----

## https://fertilizer-prediction.streamlit.app/

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```



<img width="1920" height="877" alt="Screenshot (500)" src="https://github.com/user-attachments/assets/037eec1d-1278-4f6a-85ec-de9f8b2df589" />

