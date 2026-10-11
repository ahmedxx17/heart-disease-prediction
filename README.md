# 🫀 Heart Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Course](https://img.shields.io/badge/COMSATS-AIC354-004080)](https://www.comsats.edu.pk/)

> **🔗 Live Streamlit Web App Demo:** [LIVE_DEMO_LINK_HERE](LIVE_DEMO_LINK_HERE)  
> *(Update this placeholder with your live Streamlit Community Cloud URL after deployment).*

---

## 📌 Project Overview
This project is an end-to-end Machine Learning assignment for **AIC354 (Machine Learning)** at **COMSATS University Islamabad**. 

It adapts the 8-step machine learning teaching pipeline from the reference **Titanic Survival Prediction** project into a clinically meaningful **Heart Disease Prediction System** using the benchmark **UCI Cleveland Heart Disease Dataset**.

### Reference Project Structure Adherence
Following the reference repository (`https://github.com/haqnawaz99/ml-teaching-notebooks`):
- All notebook code, data files, and trained model artifacts reside in `1 - Heart Disease Prediction/`.
- The deployed web application resides in `streamlit-app/`.
- Root contains `requirements.txt`, `runtime.txt`, and project documentation (`README.md`).
- Follows the identical 10-step teaching flow: **Import Libraries $\rightarrow$ Load Data $\rightarrow$ Preprocess $\rightarrow$ Feature Extraction $\rightarrow$ Encode $\rightarrow$ Train $\rightarrow$ Test $\rightarrow$ Application $\rightarrow$ Feedback $\rightarrow$ Deploy Web App**.

---

## 📊 Dataset & Features
- **Data Source:** [UCI Machine Learning Repository – Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Data File:** `processed.cleveland.data` (303 instances, 14 raw attributes).
- **Target Transformation:** The raw target attribute `num` (0–4) is binarized:
  - `0`: Absence of coronary artery disease (**No Heart Disease**)
  - `1, 2, 3, 4`: Presence of coronary artery disease ($\ge 50\%$ stenosis) (**Heart Disease Present**)

### 🔬 4 Selected Features (Constraint Fulfillment)
Per the assignment constraint, exactly **4 input features** were selected:

| Feature | Type | Description | Values / Range | Clinical Justification |
| :--- | :--- | :--- | :--- | :--- |
| **`Age`** | Numerical | Patient age in years | 29 – 77 years | Core non-modifiable demographic risk factor; cardiovascular risk increases progressively with age. |
| **`Gender`** | Categorical | Biological sex | `Female`, `Male` | Well-documented sex disparities in cardiovascular pathology and ischemic risk profile. |
| **`ChestPain`** (`cp`) | Categorical | Chest pain classification | `Typical Angina`, `Atypical Angina`, `Non-Anginal Pain`, `Asymptomatic` | Primary symptomatic diagnostic marker for coronary artery disease. Asymptomatic ischemia is prevalent in high-risk patients. |
| **`MaxHeartRate`** (`thalach`) | Numerical | Peak exercise heart rate | 71 – 202 bpm | Reflects chronotropic competence and cardiac functional reserve under exercise stress testing. |

#### Why These 4 Features?
1. **Balanced Mix:** Combines 2 continuous numerical features (`Age`, `MaxHeartRate`) and 2 categorical features (`Gender`, `ChestPain`).
2. **High Clinical Correlation:** Supported by cardiology diagnostic protocols for non-invasive risk stratification.
3. **Zero Missing Values:** In the Cleveland dataset, missing values (`?`) only occur in `ca` (4 missing) and `thal` (2 missing). Our 4 selected features have **0 missing values across all 303 patients**, ensuring clean training without imputation noise.

---

## 📈 Model Performance & Evaluation
- **Algorithm:** Support Vector Classifier (`SVC(kernel='linear', probability=True, random_state=42)`)
- **Data Split:** 80% Training (242 instances), 20% Held-Out Testing (61 instances), stratified.

| Metric | Score | Clinical Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **~75.4%** | 46 out of 61 unseen patients correctly classified. |
| **Precision** | **0.78** | When the model predicts heart disease, it is correct 78% of the time. |
| **Recall (Sensitivity)** | **0.64** | Detects 64% of true positive disease cases in the test set. |
| **F1-Score** | **0.76** | Balanced harmonic mean of precision and recall. |

### Confusion Matrix
```
                     Predicted: No    Predicted: Disease
Actual: No Disease        28                  5
Actual: Disease           10                 18
```

---

## 📂 Directory Structure
```
heart-disease-prediction/
├── 1 - Heart Disease Prediction/
│   ├── Heart_Disease_Prediction.ipynb    # Main student Jupyter notebook (runs top-to-bottom)
│   ├── processed.cleveland.data          # Raw UCI dataset (303 records, 14 attributes)
│   ├── sample-data.csv                   # Cleaned dataset (4 features + binary target)
│   ├── sample-data-encoded.csv           # Fully label-encoded dataset
│   ├── sample-data-encoded-output.csv    # Encoded output dataset
│   ├── training-data-encoded.csv         # 80% training set
│   ├── testing-data-encoded.csv          # 20% testing set
│   ├── model-predictions.csv             # Test predictions vs ground truth
│   └── heart_model.pkl                   # Trained Support Vector Classifier artifact
├── streamlit-app/
│   ├── app.py                            # Streamlit web application
│   ├── heart_model.pkl                   # Trained model copy for deployment
│   └── requirements.txt                  # Pinned dependencies for Streamlit Cloud
├── requirements.txt                      # Root dependencies
├── runtime.txt                           # Python version (3.11)
└── README.md                             # Complete project documentation
```

---

## 💻 Local Setup & Execution

### 1. Clone or Open the Repository
```bash
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Create and Activate Virtual Environment
**Windows:**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook
```bash
jupyter notebook "1 - Heart Disease Prediction/Heart_Disease_Prediction.ipynb"
```
Select **Kernel $\rightarrow$ Restart & Run All** to run all cells top-to-bottom.

### 5. Run the Streamlit Web Application Locally
```bash
streamlit run streamlit-app/app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

Follow these step-by-step instructions to deploy your live web application for assignment grading:

### Step 1: Create a GitHub Repository
1. Log in to [GitHub](https://github.com).
2. Click **New Repository** (top-right).
3. Name your repository (e.g., `heart-disease-prediction`), keep it **Public**, and click **Create repository**.
4. In your terminal, initialize git and push:
   ```bash
   git init
   git add .
   git commit -m "Complete Heart Disease Prediction assignment"
   git branch -M main
   git remote add origin https://github.com/<your-username>/heart-disease-prediction.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **Create app** $\rightarrow$ **Deploy a public app from GitHub**.
3. Fill in the fields:
   - **Repository:** `<your-username>/heart-disease-prediction`
   - **Branch:** `main`
   - **Main file path:** `streamlit-app/app.py`
4. Click **Deploy**.
5. Wait ~1-2 minutes for dependencies to install. Once live, you will receive a public URL (e.g., `https://<your-username>-heart-disease.streamlit.app`).

### Step 3: Update Submission Links
1. Copy your live `.streamlit.app` URL.
2. In `1 - Heart Disease Prediction/Heart_Disease_Prediction.ipynb`, edit the very first cell:
   Replace `LIVE_DEMO_LINK_HERE` with your live URL.
3. In this `README.md`, replace `LIVE_DEMO_LINK_HERE` with your live URL.
4. Commit and push the update:
   ```bash
   git commit -am "Update live deployment demo URL"
   git push
   ```

---

## 🎓 Viva & Defense Questions (Exam Preparation)

### Q1: Why use a Support Vector Classifier (SVC) for this task?
> **Answer:** Support Vector Classifiers find the maximum-margin hyperplane separating positive and negative classes. For small-to-medium clinical datasets (like Cleveland's 303 rows), SVMs have strong theoretical guarantees against overfitting and perform reliably in multidimensional spaces compared to deep neural networks.

### Q2: Why choose a Linear kernel over an RBF kernel?
> **Answer:** With 4 input features and a limited dataset, a linear kernel provides high interpretability, lower variance, and fewer hyperparameters to tune ($C$ alone, without needing $\gamma$). Empirically, linear SVC achieved 75.4% test accuracy compared to 70.5% with the non-linear RBF kernel on this feature subset.

### Q3: Why Label Encoding instead of One-Hot Encoding?
> **Answer:** To preserve exact structural parity with the reference Titanic teaching notebook (which used `LabelEncoder` for `Gender` and `Embarked`). Furthermore, `LabelEncoder` outputs single-column numeric indices that match the 4-input feature vector schema expected by the SVC and the deployed app widgets.

### Q4: How is probability calculated in an SVC?
> **Answer:** Standard SVMs output signed geometric distances to the decision hyperplane, not calibrated probabilities. Setting `probability=True` enables Platt scaling—fitting a logistic regression model over SVM decision scores using 5-fold cross-validation—which produces calibrated disease probability scores ($P(y=1|X)$).

### Q5: Why is Recall critical in medical applications?
> **Answer:** False negatives in healthcare are dangerous: missing an individual who has coronary disease may lead to untreated myocardial infarction. While precision measures how many predicted patients truly have disease, high recall ensures that at-risk patients are not missed during preliminary screening.

---

## 📝 Manual Submission Checklist (Google Classroom)
Before submitting your assignment, verify:
- [ ] Created GitHub repository and pushed all code and files.
- [ ] Deployed web app on Streamlit Community Cloud and confirmed it opens publicly.
- [ ] Replaced `LIVE_DEMO_LINK_HERE` in the first cell of `Heart_Disease_Prediction.ipynb`.
- [ ] Replaced `LIVE_DEMO_LINK_HERE` in `README.md`.
- [ ] Ran the notebook from top-to-bottom in Jupyter (`Kernel -> Restart & Run All`) with 0 errors.
- [ ] Uploaded repository link and live app link to Google Classroom according to course guidelines.
