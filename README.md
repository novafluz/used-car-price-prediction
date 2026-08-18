# 🚗 Used Car Price Prediction

A Machine Learning project that predicts the price of a car based on various vehicle features such as brand, model, model year, milage, fuel type, engine, transmission, exterior & interior color, accident history and clean title.

---

## Dataset

This project uses the [Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset) provided by Taeef Najib on Kaggle. Please refer to the dataset page for licensing terms and conditions.

## Project Structure

```text
used-car-price-prediction/
│
├── data/
├── fig/ # figures and visualizations
├── notebooks/
│   └── exploration.ipynb # explanatory data analysis
│
├── src/
│   ├── application.py
│   ├── data_cleaning.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── loadData.py
│   ├── preprocessing.py
│   └── train.py
│
├── LICENSE
├── README.md
├── main.py
└── requirements.txt
```

## Machine Learning Pipeline

#### 1. Exploratory Data Analysis

The dataset was explored to understand the distribution of the data, missing values, categorical features, numerical features, and the relationship between vehicle features and car prices.

#### 2. Data Loading

The dataset is loaded from the `data/` directory using pandas.

#### 3. Data Cleaning

The raw data is cleaned by handling missing values, removing unnecessary features, and converting some features into suitable formats for further processing.

#### 4. Feature Engineering

New features are created from the existing vehicle information to provide more useful information for the machine learning models.

#### 5. Data Preprocessing

Numerical and categorical features are processed separately. Numerical features are scaled when necessary, while categorical features are one-hot encoded before being passed to the models.

#### 6. Model Training

Three regression models were trained and compared:

* Linear Regression
* Random Forest
* XGBoost

#### 7. Model Evaluation

The models are evaluated on the test set using R², RMSE, and MAE.

## Results

The performance of the three models on the test set is shown below:

| Model             |         R² |          RMSE |          MAE |
| ----------------- | ---------: | ------------: | -----------: |
| Linear Regression |     0.7037 |     29,479.33 |    14,461.17 |
| Random Forest     |     0.7426 |     27,476.76 |     9,992.22 |
| XGBoost           | **0.8238** | **22,736.23** | **9,533.42** |

Among the three models, **XGBoost** achieved the best performance with an R² score of **0.8238**, an RMSE of **22,736.23**, and an MAE of **9,533.42** on the test set.

## ⚙️ Installation

1. Clone the repository

   ```bash
   git clone https://github.com/novafluz/used-car-price-prediction.git
   ```

2. Create and activate a conda environment

   ```bash
   conda create -n carprice python=<your_python_version>
   conda activate carprice
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Train the models

   ```bash
   python src/train.py
   ```

2. Evaluate the models

   ```bash
   python src/evaluate.py
   ```

3. Run the application

   ```bash
   python main.py
   ```

## License

This project is licensed under the MIT License.
