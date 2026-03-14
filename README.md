# **Steam Video Game Recommender System**

Recommendation system for video games using content-based, collaborative filtering, and hybrid models.
The project reproduces the full machine learning pipeline: data preprocessing, feature engineering, model training, and evaluation.

---

## **Project Overview**

**Goal**: recommend video games to users based on game characteristics and user behavior.

The system explores three approaches:

* **Content-based filtering** – recommends games similar to those previously liked.
* **Collaborative filtering** – recommends games liked by similar users.
* **Hybrid model** – combines both methods to improve recommendation quality.

---

## **Repository Structure**

**system_recommendation_steam/**

**notebooks/**
│
├── 01_information.ipynb
├── 02_preprocessing.ipynb
├── 03_eda.ipynb
├── 04_feature_engineering.ipynb
├── 05_prepare_data.ipynb
├── 06_base_model.ipynb
├── 07_content_filtering.ipynb
├── 08_collaborative_filtering.ipynb
├── 09_hybrid.ipynb
└── 10_evaluation.ipynb

**src/**
├── evaluation_functions.py
├── recommendation_functions.py
└── recommender_utils.py

**reports/**
HTML exports of the notebooks for quick visualization

**docs/**
TFM_Alonso_Gil_Laura.pdf

---

## **Dataset**

Steam video game dataset containing:
* game metadata
* genres and tags
* user reviews and interactions

The dataset is not included in this repository due to size constraints.

Place the dataset inside:

data/

after downloading it.

---

## **Pipeline**

The project follows a structured ML workflow.

1. **Information**. Extraction of raw Steam data from public APIs and storage in a SQLite database in order to build the initial database schema.

2. **Preprocessing**. Initial quality filtering (minimum number of reviews per video game and minimum number of games per user) and generation of the cleaned tables that will be used in the following steps.


3. **Exploratory Data Analysis (EDA)**. Exploratory data analysis of the dataset (structure, missing values, duplicates, genres, boolean features and text variables).
This step defines the feature engineering decisions applied in the next script.

4. **Feature Engineering (FE)**. Implementation of the final feature engineering for games:
* Variable selection and removal
* Creation of developer_level and dummy variables
* Construction of text_full_clean
* Creation of the structured feature base (scaled matrix)
* Creation of the textual feature base (TF-IDF matrix)

These matrices are used as input for the recommendation models.

5. **Perpare data**. Alignment and cleaning of all final tables, filtering of games and users without complete information, and generation of the train/test split using a Leave-3-Out (LMO) evaluation scheme.

6. **Base model**. Construction and evaluation of a baseline popularity recommender based on positive_reviews. Evaluation metrics are calculated for k = 10 and k = 50 under the Leave-3-Out (LMO) scheme.

7. **Content filtering**. Construction and evaluation of the content-based recommendation system. A content profile is generated for each user using the structured and textual matrices.
Different weighting combinations are tested and evaluation metrics are computed for k = 10 and k = 50 under the LMO scheme.

8. **Collaborative filtering**. Construction and evaluation of the collaborative filtering recommendation system. Using the normalized user–game interaction matrix, a k-Nearest Neighbors (k-NN) model is trained. Evaluation metrics are computed for k = 10 and k = 50 under the LMO scheme.

9. **Hybrid**. Combination of the content-based and collaborative scores in a hybrid recommendation system. Different values of α (alpha) are evaluated to balance both models, selecting the optimal weight and computing evaluation metrics for k = 10 and k = 50 under the LMO scheme.

10. **Evaluaton**. Loading and analysis of the evaluation metrics for the popularity, content-based, collaborative, and hybrid systems for k = 10 and k = 50. The step compares the quantitative performance of the four approaches and provides a qualitative analysis through case studies.

---

## **Installation**
Clone the repository:
* git clone https://github.com/<username>/system_recommendation_steam.git

Install dependencies:
* pip install -r requirements.txt

---

## **Running the Project**

Execute notebooks in order:

* 01_information 
* 02_preprocessing 
* 03_eda 
* 04_feature_engineering 
* 05_prepare_data 
* 06_base_model 
* 07_content_filtering 
* 08_collaborative_filtering 
* 09_hybrid 
* 10_evaluation