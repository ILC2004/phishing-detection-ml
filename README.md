# Phishing Detection Prototype

A simple dissertation prototype for phishing detection using **Python, scikit-learn, and Streamlit**.

---

## What the system does

- Accepts **email text or a URL** as input  
- Predicts whether the message is **Legitimate** or **Phishing**  
- Displays a **confidence score** for the prediction  
- Shows several **basic indicators/features** explaining the model’s decision  

This prototype demonstrates how machine learning can be used to automatically identify phishing emails based on textual patterns.

---

## Project structure

```
app.py                     Streamlit user interface
train_model.py             Model training script
model_utils.py             Feature extraction and preprocessing functions
evaluate_model.py          Model evaluation and performance graphs
explain_model.py           Feature importance analysis
phishing_pipeline.joblib   Saved trained machine learning model
final_phishing_dataset.csv Cleaned dataset used for training
requirements.txt           Python dependencies
README.md                  Project documentation
```

---

## Running the prototype (Windows)

Open **PowerShell** inside the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Then open the application in your browser:

```
http://localhost:8501
```

---

## Using the prototype

1. Paste an email message or URL into the input field  
2. Click **Analyse**  
3. The system will display:
   - Prediction (**Legitimate or Phishing**)  
   - Confidence score  
   - Basic indicators explaining the prediction  

---

## Dataset information

The model was trained using a cleaned dataset containing:

- **137,721 total email messages**
- **72,289 phishing emails**
- **65,432 legitimate emails**

Due to file size limitations, the full dataset (`final_phishing_dataset.csv`) is **not included in the submission**.

Instead, a **small dataset sample** is provided to demonstrate the dataset structure.

---

## Model evaluation

The trained model achieved the following performance:

| Metric | Result |
|------|------|
| Accuracy | 97.6% |
| Precision | 97.3% |
| Recall | 98.1% |
| F1-score | 0.977 |
| ROC-AUC | 0.997 |

Evaluation outputs included in the submission:

```
confusion_matrix.png
roc_curve.png
precision_recall_curve.png
classification_report.txt
metrics.json
```

These files demonstrate the performance of the phishing detection model.

---

## Recreating the Python environment

The Python virtual environment (`.venv`) is **not included in the submission** because it contains installed packages and significantly increases the project size.

The environment can be recreated using the dependencies listed in `requirements.txt`.

Run:

```
pip install -r requirements.txt
```

This will install all required libraries.

---

## Demonstration for the dissertation

During the prototype demonstration:

1. Launch the Streamlit application  
2. Paste an example **phishing message**  
3. Paste an example **legitimate email**  
4. Show the prediction and confidence score  
5. Display the feature explanation panel  

This demonstrates the **complete machine learning pipeline**, from input text to phishing prediction.

---

## Project purpose

This prototype demonstrates how **machine learning techniques can be applied to detect phishing emails automatically**, helping identify suspicious messages based on linguistic patterns found in phishing attacks.