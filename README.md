
## Used Car Price Prediction

This project aims to develop a machine learning model that predicts the price of a used car based on its specifications and other relevant features. Users can estimate a fair market price for their desired car. This tool can help buyers make informed decisions and negotiate better deals in the used car market.

I found this dataset from kaggle which contains around `37813` rows and `66`columns from **car dekho website**.

I performed Data cleaning, Exploratory data analysis (EDA), data transformation and model building.

### Dataset link

[Used car price prediction dataset - Kaggle](https://www.kaggle.com/datasets/sukritchatterjee/used-cars-dataset-cardekho)

### Feature Selection

I selected the following features for price prediction:

- `manufacturer` - label encoded
- `model` - Target Encoded with mean `listing_price`
- `km_driven` 
- `Insured`
- `seller` - Dealer or individual
- `owner` - First, second, third, fourth and fifth
- `car_age` - car age from the date of manufacture 
- `features_count` - originally features count is a top_features, which is a array of top features in a car i.e. no. of features, due to many unique values I created this features-count.
- `state`- One hot encoded states.

These features were found to be useful during prediction.

### Model Selection

I experimented with several **ensemble models**, and **XGBoost** performed the best.  
XGBoost was trained using **GPU**, and predictions are performed on **CPU** to avoid errors.
