# Modularized EV Battery Project Plan

Instead of a monolithic single-file setup, we will decouple your Machine Learning pipeline into distinct, industry-standard architectural layers. We will also produce professional presentation material.

## Proposed Changes

### 1. Folder Structure Setup
We will establish a proper directory tree directly inside your workspace:
- `/data/`: To store raw and processed datasets.
- `/models/`: To hold serialized (`.pkl`) trained models.
- `/src/`: For backend model training and data preprocessing scripts.
- `/app/`: The interactive front-end Streamlit web application.

---

### 2. File Restructuring

#### [DELETE] `app.py`
We will remove the monolithic script we previously generated to make way for the modular version.

#### [NEW] `requirements.txt`
Dependencies list: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`.

#### [NEW] `src/data_processing_and_training.py`
This script will perform the heavy lifting:
- Load the metadata from `/data/nasa_battery_dataset.csv` and merge it with the raw sensor files located in `/data/raw_sensor_data/`
- Conduct strict Pandas imputation and Scikit-Learn `StandardScaler` transformations.
- Complete the 80/20 train-test splits.
- Train the 3 required models (Multiple Linear Regression, Random Forest, XGBoost).
- Evaluate based on MAE, MSE, RMSE, R-Squared.
- Save the winning `.pkl` model into the `/models/` directory.
- Critically, this script will ALSO save a `training_metadata.pkl` file into `/models/`. This metadata will contain the extracted loss trajectories, feature weights, and residual scatter mappings so your Streamlit app can display these charts instantly without retraining the model.

#### [NEW] `app/app.py`
The UI file will:
- Load the pre-trained `best_model.pkl` and `training_metadata.pkl` from `/models/`.
- Draw the 4 requested visualisations (Degradation curve, Loss Graph, Feature Importance, Residuals).
- Serve the interactive sidebar taking inputs for live capacity prediction inference.

---

### 3. Creating Deliverables 3 & 4 (Speech & Q&A)

#### [NEW] `presentation_and_qa.md` (Artifact)
We will generate a markdown file comprehensively mapping out:
- **The Expert Presentation Script**: An introduction, architectural analogy, and model strategy written in plain, compelling English.
- **Teacher Q&A Cheat Sheet**: Direct, simple answers on data scaling, R-Squared meaning, RMSE significance, Model communication via `.pkl`, and the 80/20 split philosophy.

## User Review Required

> [!NOTE]
> Please review this structure. To display the four requested visualization graphs cleanly in `app/app.py` without forcing Streamlit to retrain the models every time the page refreshes, we will save the graph data points during the backend `src/` training phase. Streamlit will simply load these points for display. This guarantees blazing-fast UI performance mapping to enterprise best-practice. Let me know if you approve this approach!

## Verification Plan

### Automated Tests
- Running `python src/data_processing_and_training.py` physically outputs files to `/data/` and `/models/`.
- Running `streamlit run app/app.py` starts the web server locally without runtime error.

### Manual Verification
- Verify the presentation script flows smoothly in conversational English while covering all technical milestones accurately.
- Verify the Streamlit app loads the graphs properly utilizing only pre-trained `.pkl` artifacts.
