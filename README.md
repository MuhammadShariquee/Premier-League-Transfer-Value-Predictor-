#  Premier League Transfer Value Predictor

A **Data Science & Machine Learning regression project** that estimates a football player's transfer fee using age, position, playing time, attacking output, passing, progression, shooting, and defensive performance statistics.

The project combines **Transfermarkt transfer records** with **FBref player-performance data** and uses a **Random Forest Regressor** to learn patterns between player performance and actual transfer fees.

> **Important:** This project does not claim that transfer fees are determined purely by on-field statistics. Real-world transfer prices are also influenced by factors such as contract length, club finances, market conditions, player reputation, release clauses, and negotiation context. The model only learns patterns from the available dataset.

---

## 📌 Project Overview

Football transfer fees can vary significantly between players, even when their performance statistics appear similar.

This project investigates:

* How transfer data can be cleaned and prepared for analysis
* How player-performance data can be matched with transfer records
* Which performance variables contain useful predictive information
* How well a Random Forest regression model can estimate transfer fees
* How model performance compares with a simple mean-value baseline
* Which features the final model relies on most according to permutation importance

### Objective

Build a regression model that predicts a player's transfer fee in euros from their:

* Age
* Position
* Playing time
* Goals and assists
* Expected goals and assists
* Shooting statistics
* Passing statistics
* Progressive actions
* Defensive statistics
* Successful take-ons

---

# 📊 Data Sources

## Transfermarkt

The project initially inspected transfer datasets covering:

**2019–2025**

The final modeling task focuses on the **2025 Premier League incoming transfer window**.

The following records were retained:

* Premier League transfers
* Incoming transfers
* Permanent transfers
* Transfers with a known fee

This produced:

**138 valid transfer records**

before matching them with FBref.

## FBref

The project uses the lighter FBref player-performance dataset:

* **2,854 player observations**
* **165 columns**

The lighter dataset was selected because it contained the required player-performance statistics without the additional duplicated metadata found in the larger dataset.

---

# 🔄 Data Preparation Workflow

The project follows this general pipeline:

```text
Transfermarkt Data
       │
       ▼
Filter 2025 Premier League
       │
       ▼
Keep Incoming Permanent Transfers
       │
       ▼
Remove Missing Transfer Fees
       │
       ▼
138 Transfer Records
       │
       │
       ├──────────────┐
       │              │
       ▼              ▼
Transfermarkt       FBref
Player Data          Performance Data
       │              │
       └──────┬───────┘
              ▼
       Player Name Matching
              │
              ▼
       Verified Name Corrections
              │
              ▼
       91 Matched Players
              │
              ▼
       Feature Selection
              │
              ▼
       Machine Learning
              │
              ▼
       Random Forest Regressor
              │
              ▼
       Transfer Fee Prediction
```

---

# 🧹 Data Cleaning & Player Matching

Direct name matching initially identified **87 players**.

Some players had differences between the Transfermarkt and FBref names, such as:

* Accents
* Punctuation
* Minor spelling differences

Four verified name mappings were explicitly corrected:

```text
Dário Essugo  → Dario Essugo
Hugo Ekitiké  → Hugo Ekitike
James McAtee  → James Mcatee
Yéremy Pino   → Yeremi Pino
```

After these verified corrections:

**91 players were successfully matched.**

Some players appeared multiple times in FBref because they played for multiple clubs or competitions. Their selected performance statistics were therefore aggregated at player level.

Final modeling dataset:

**91 players × 33 modeling columns**

---

# 🧠 Machine Learning Approach

This is a **supervised regression problem**.

The target variable is:

```text
Transfer Fee (€)
```

Because the target is a continuous numerical value, the project uses **Random Forest Regression**.

### Why Random Forest?

Random Forest was selected because it can:

* Model nonlinear relationships
* Capture interactions between features
* Handle different numerical feature scales
* Work well with a mixture of player-performance variables
* Avoid requiring a simple linear relationship between inputs and transfer fee

---

# ⚙️ Feature Engineering

The final reduced model uses 18 features:

### Player Profile

* `Age`
* `Pos`

### Playing Time

* `Min`

### Attacking

* `Gls`
* `Ast`
* `xG`
* `xAG`

### Shooting

* `Sh`
* `SoT`

### Creativity & Progression

* `KP`
* `PrgP`
* `PrgC`
* `PrgR`

### Defensive Performance

* `Succ`
* `TklW`
* `Tkl`
* `Int`
* `Clr`

`Pos` is categorical and is converted using **One-Hot Encoding**.

The preprocessing and model are combined in a Scikit-learn `Pipeline` so that the same transformations are consistently applied during training and prediction.

---

# 🧪 Model Development

Several experiments were performed.

| Model                      |         MAE |        RMSE |         R² |
| -------------------------- | ----------: | ----------: | ---------: |
| Mean Baseline              |     €20.06M |     €26.59M |    -0.0657 |
| Initial Random Forest      |     €13.58M |     €18.54M |     0.4670 |
| Random Forest + Log Target |     €14.07M |     €20.62M |     0.3373 |
| Reduced Random Forest      |     €13.36M |     €18.52M |     0.4692 |
| **Tuned Random Forest**    | **€12.97M** | **€17.83M** | **0.5085** |

The reported results above are from **5-fold cross-validation**.

---

# 🔧 Hyperparameter Tuning

`GridSearchCV` was used to test different Random Forest configurations.

The search evaluated:

* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`

A total of:

**72 parameter combinations × 5 folds = 360 fits**

were evaluated.

The selected configuration was:

```python
{
    "max_depth": 5,
    "max_features": 0.7,
    "min_samples_leaf": 1,
    "min_samples_split": 5
}
```

---

# 📈 Final Cross-Validation Results

The tuned Random Forest achieved:

```text
MAE  : €12,973,289 ± €2,795,792
RMSE : €17,833,555 ± €4,306,002
R²   : 0.5085 ± 0.1666
```

Individual fold R² scores were:

```text
0.7682
0.4581
0.2531
0.4966
0.5665
```

The variation between folds highlights the difficulty of predicting transfer fees from a relatively small dataset.

---

# 🧪 Holdout Evaluation

A separate 20% holdout split was also evaluated.

```text
MAE  : €9,126,053
RMSE : €11,374,137
R²   : 0.7764
```

These results should be interpreted carefully.

The hyperparameters were selected using cross-validation before the final holdout evaluation. Therefore, this holdout score is best treated as an **illustrative project evaluation**, rather than a publication-grade independent estimate.

The 5-fold cross-validation results provide a more useful indication of expected generalization within this workflow.

---

# 🔍 Feature Importance

Permutation importance was used to investigate which features most affected model performance.

The strongest features in the final run included:

| Feature                      | Importance |
| ---------------------------- | ---------: |
| Successful Take-ons (`Succ`) |     0.1451 |
| Shots on Target (`SoT`)      |     0.1095 |
| Age                          |     0.0857 |
| Shots (`Sh`)                 |     0.0596 |
| Expected Goals (`xG`)        |     0.0504 |
| Goals (`Gls`)                |     0.0412 |
| Progressive Passes (`PrgP`)  |     0.0197 |
| Key Passes (`KP`)            |     0.0139 |

Permutation importance measures how model performance changes when a feature is shuffled.

Therefore, these values should be interpreted as **model associations**, not proof that a particular statistic causes transfer prices to increase or decrease.

---

# ⚽ Example Prediction

The notebook includes a reusable prediction function.

For example, providing a player's performance statistics such as:

```text
Age: 23
Position: DF
Minutes: 2766
Goals: 4
Assists: 1
xG: 2.3
xAG: 0.2
Shots: 16
Shots on Target: 7
Key Passes: 3
Progressive Passes: 90
Progressive Carries: 15
Progressive Receives: 3
Successful Take-ons: 9
Tackles Won: 18
Tackles: 31
Interceptions: 27
Clearances: 99
```

produced an example prediction of approximately:

# **€26.40 million**

This demonstrates how the trained model can be used programmatically to estimate a transfer fee for a new player profile.

---

# 📁 Repository Structure

```text
Premier-League-Transfer-Value-Predictor/
│
├── Premiere League Tranfer Value/
│   │
│   ├── 2019.csv
│   ├── 2020.csv
│   ├── 2021.csv
│   ├── 2022.csv
│   ├── 2023.csv
│   ├── 2024.csv
│   ├── 2025.csv
│   │
│   ├── players_data_light-2024_2025.csv
│   ├── players_data-2024_2025.csv
│   │
│   └── PremierLeagueTransferValuePredictorCleanDocumented.ipynb - Colab.pdf
│
└── README.md
```

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Google Colab**
* **Jupyter Notebook**
* **GitHub**

### Machine Learning

* Random Forest Regression
* Dummy Regression Baseline
* One-Hot Encoding
* Scikit-learn Pipelines
* Cross-Validation
* GridSearchCV
* Permutation Importance
* MAE
* RMSE
* R²

---

# ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/MuhammadShariquee/Premier-League-Transfer-Value-Predictor-.git
```

### 2. Open the notebook

Open:

```text
PremierLeagueTransferValuePredictorCleanDocumented.ipynb
```

in Google Colab or Jupyter Notebook.

### 3. Upload the datasets

Make sure the required CSV files are available in the notebook environment.

### 4. Run the notebook from top to bottom

The notebook performs:

```text
Data Loading
    ↓
Data Inspection
    ↓
Data Cleaning
    ↓
Player Matching
    ↓
Feature Selection
    ↓
Exploratory Analysis
    ↓
Baseline Model
    ↓
Random Forest
    ↓
Model Experiments
    ↓
Hyperparameter Tuning
    ↓
Cross-Validation
    ↓
Holdout Evaluation
    ↓
Feature Importance
    ↓
Individual Prediction
```

---

# ⚠️ Limitations

This project has several important limitations.

### 1. Small modeling dataset

Only **91 players** were successfully matched between the 2025 transfer records and FBref performance data.

### 2. Performance statistics are not the complete transfer market

Transfer fees can depend on many factors that are not included in the model, including:

* Contract length
* Release clauses
* Club finances
* Negotiation strategy
* Player reputation
* Market conditions
* Remaining contract years
* Competition between buying clubs

### 3. Transfer fees are highly skewed

The dataset contains a small number of very expensive transfers alongside many lower-value transfers.

The maximum observed fee in the final dataset was:

**€145 million**

### 4. Limited season coverage for final modeling

Although transfer datasets from 2019–2025 were inspected, the final model was built using the matched **2025 Premier League transfer dataset**.

### 5. Model evaluation uncertainty

The relatively small sample means that performance can vary substantially between cross-validation folds.

Therefore, the reported scores should not be interpreted as proof that the model will achieve the same accuracy on future transfer windows.

---

# 🚀 Future Improvements

Possible improvements include:

* Expand the modeling dataset across multiple seasons
* Include more leagues
* Add contract-length information
* Include player market value
* Include club financial information
* Include player reputation or international experience
* Test additional regression algorithms
* Use nested cross-validation
* Compare against stronger baselines
* Deploy the predictor as a web application
* Build an interactive player-statistics input interface
* Add automated data collection and preprocessing

---

# 📚 Project Takeaway

This project demonstrates an end-to-end **Data Science and Machine Learning workflow**:

```text
Raw Data
   ↓
Data Cleaning
   ↓
Data Integration
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Baseline
   ↓
Machine Learning
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Feature Interpretation
   ↓
Prediction
```

The project shows how real-world football data from different sources can be cleaned, matched, transformed, and used to build a machine-learning regression model.

---

## 👨‍💻 Author

**Muhammad Sharique**

BS Software Engineering Student
University of Sindh, Jamshoro

GitHub: [MuhammadShariquee](https://github.com/MuhammadShariquee)

---

⭐ If you found this project interesting, feel free to explore the notebook and the data-processing workflow.
