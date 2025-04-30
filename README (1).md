# Heart-disease-prediction-model

This project is a machine learning-based web application that predicts the likelihood of heart disease in a person based on key health parameters. It uses a trained classification model deployed using Streamlit for interactive use.

🚀 Features
	•	Predicts risk of heart disease using clinical features
	•	User-friendly interface with Streamlit
	•	Trained on a public heart disease dataset
	•	Quick and efficient real-time predictions

📁 Files Overview
	•	HeartDiseasePrediction.ipynb: Jupyter notebook containing EDA, preprocessing, model training, and evaluation.
	•	dataset_heart.csv: Dataset used for training and testing the machine learning model.
	•	streamlit2.py: Streamlit web application script for model deployment.

🛠️ How to Run
	1.	Clone the repository
                                    git clone https://github.com/your-username/heart-disease-prediction.git
                                    cd heart-disease-prediction
                                    
  2.  Install dependencies          pip install -r requirements.txt
      
  3.  Run the app                   streamlit run streamlit2.py

  🧠 Model Information

The model was trained using standard classifiers such as Logistic Regression, Decision Trees, and Random Forest on a cleaned version of the heart disease dataset.

📊 Dataset Features
	•	Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, etc.
	•	Target: Presence (1) or absence (0) of heart disease.

  🧠 How It Works

1. User enters medical details in the app (e.g., age, cholesterol, resting BP).
2. Inputs are passed to the trained ML model.
3. The model predicts the likelihood of heart disease.
4. The result is displayed instantly.

📌 Requirements
	•	Python 3.7+
	•	Streamlit
	•	Pandas, Numpy
	•	Scikit-learn
	•	Matplotlib / Seaborn (for notebook)

 This project is open-source and available under the MIT License.
