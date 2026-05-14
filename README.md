# Phishing Detection System Using Machine Learning

The system uses machine learning to classify email text or URLs as either **Legitimate** or **Phishing** through a simple Streamlit interface.

---

## Project overview

This project demonstrates how machine learning can support phishing detection by analysing textual patterns found in emails and URLs. The prototype allows a user to paste a message or URL into the interface and receive:

- a phishing or legitimate prediction
- a confidence score
- simple indicators explaining why the result was produced

The aim is to make phishing detection understandable for non-technical users while still showing a complete machine learning workflow.

---

## Key features

- Email text and URL input
- Real-time phishing classification
- TF-IDF feature extraction
- Logistic Regression deployment model
- Model comparison using Logistic Regression, Linear SVM, and Random Forest
- Confidence score output
- Basic explanation indicators for user understanding
- Evaluation outputs including confusion matrix, ROC curve, and precision-recall curve

---

## Technologies used

- Python
- Streamlit
- scikit-learn
- pandas
- NumPy
- joblib
- matplotlib

---

## Project structure

```text
app.py                       Streamlit user interface
train_model.py               Trains the final model
model_utils.py               Preprocessing and feature utility functions
evaluate_model.py            Generates evaluation metrics and graphs
compare_models.py            Compares Logistic Regression, Linear SVM, and Random Forest
explain_model.py             Produces feature analysis
phishing_pipeline.joblib     Saved trained machine learning pipeline
dataset_sample.csv           Small sample of the cleaned dataset
model_comparison_results.csv Model comparison results
evaluation_outputs/          Evaluation graphs and reports
requirements.txt             Python dependencies
README.md                    Project documentation
```

The full cleaned dataset is not included in the repository because of file size limits. A sample dataset is included to show the dataset structure.

---

## Dataset information

The model was trained and evaluated using a cleaned labelled dataset containing:

- **137,721 total email messages**
- **72,289 phishing emails**
- **65,432 legitimate emails**

The dataset was split into:

- **80% training data**
- **20% testing data**

This resulted in **27,545 test samples** used for final evaluation.

---

## Model performance

The deployed Logistic Regression model achieved:

| Metric | Result |
|---|---:|
| Accuracy | 97.6% |
| Precision | 97.3% |
| Recall | 98.1% |
| F1-score | 0.977 |
| ROC-AUC | 0.997 |

Random Forest achieved slightly stronger raw performance, but Logistic Regression was selected for deployment because it provided a better balance between accuracy, speed, and interpretability.

---

## Running the prototype on Windows

Open PowerShell inside the project folder.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the Streamlit application:

```powershell
streamlit run app.py
```

Then open the local Streamlit link shown in PowerShell, usually:

```text
http://localhost:8501
```

---

## Using the prototype

1. Paste an email message or URL into the input box.
2. Click **Analyse**.
3. Review the prediction:
   - **Legitimate**
   - **Phishing**
4. Check the confidence score and explanation indicators.

---

## Evaluation outputs

The `evaluation_outputs/` folder contains evidence generated during model evaluation, such as:

```text
confusion_matrix.png
roc_curve.png
precision_recall_curve.png
classification_report.txt
metrics.json
```

These files contain evaluation metrics, graphs, and performance results used to assess the phishing detection model.

---

## Files intentionally excluded from GitHub

The following files/folders are not uploaded because they are large, generated, or machine-specific:

```text
.venv/
__pycache__/
final_phishing_dataset.csv
```

The environment can be recreated using:

```powershell
pip install -r requirements.txt
```

---

## Demonstration

For demonstration, the prototype can be shown by:

1. launching the Streamlit application
2. entering a phishing-style message
3. showing the phishing prediction and confidence score
4. entering a normal work-style message
5. showing the legitimate prediction
6. explaining how the model uses learned text patterns to classify the message

---

## Project purpose

This project demonstrates how machine learning and natural language processing can be applied to phishing detection through a real-time classification prototype and performance evaluation system.
