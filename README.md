# Lloyds Bank Customer Churn Prediction
## Executive Summary
This project, completed as part of the Lloyds Bank virtual experience programme, explored customer churn prediction using real-world-style data. Multiple datasets were integrated, ethically filtered, and analysed to uncover behavioural and demographic patterns linked to churn.

Exploratory data analysis revealed that higher login frequency correlated with retention, while churners often had slightly higher average spend but lower overall engagement. The resulting clean, balanced, and bias-conscious dataset is now ready for machine learning, with the intention to build an interpretable model to inform targeted retention strategies.
## Overview
This project aimed to identify customers at risk of churning and recommend data-driven retention strategies. The work focused on developing a supervised classification model capable of predicting churn based on demographic and behavioural factors.

The first phase, documented here, covered exploratory data analysis (EDA) and data preprocessing, laying the groundwork for accurate and ethical predictive modelling.
## Project Objectives
- Integrate disparate datasets containing customer demographics, transactions, service interactions, online activity, and churn status into a single analytical table.
- Select variables that are predictive and ethically sound, avoiding features that could introduce socio-economic or demographic bias.
- Explore, clean, and transform the data to ensure suitability for machine learning algorithms.
- Identify patterns and potential predictors of churn to inform the modelling stage.
## Data Sources
The dataset, supplied as a Microsoft Excel file, included:
1. `Customer_Demographics` - age, gender, marital status, income level.
2. `Transaction_History` - spending amounts, purchase frequency
3. `Customer_Service` - interaction types, resolution outcomes
4. `Online_Activity` - service usage channels, login frequency
5. `Churn_Status` - binary label indicating churn

The first four sources provided features, while the final table served as the target variable.
## Methodology
1. **Data Integration**
   - Aggregated each table to the customer level (e.g., total purchases, average spend, resolution rate) to create meaningful features.
   - Used `Customer_Demographics` as the base table, with left joins ensuring all 1,000 customers were retained.
   - Removed irrelevant metadata such as IDs and raw timestamps.
2. **Variable Selection**
   - Retained features such as IncomeLevel, AmountSpent, ServiceUsage, LoginFrequency, and ResolutionRate for their predictive and interpretable qualities.
   - Excluded sensitive attributes (e.g., gender, marital status) to prevent bias.
   - Removed highly correlated variables to reduce redundancy.
3. **Exploratory Data Analysis**
   - Numerical features displayed largely uniform or normal distributions; `AvgSpend` showed notable outliers.
   - Bivariate analysis revealed:
     - Churners tended to have higher average spend.
     - Lower login frequency was associated with higher churn.
     - Churners were marginally older.
   - Service channel preferences varied slightly, with mobile app users showing a higher churn rate than website users.
   - The target variable had moderate class imbalance (20.4% churners).
4. **Data Cleaning and Preprocessing**
   - Filled missing interaction-related fields with zeros, reflecting absence of contact.
   - Removed outliers in `AvgSpend` to improve balance.
   - Standardised numerical variables and one-hot encoded categorical variables, dropping reference categories to avoid multicollinearity.
   - Converted `TotalInteractions` to a binary feature for simplicity.
  
## Key Insights from EDA
- **Engagement matters**: Customers logging in more frequently were less likely to churn.
- **Spending patterns are mixed**: Churners tended to have higher average spend but lower overall spending activity.
- **Service usage differences are subtle**: Mobile app users showed marginally higher churn rates than other channels.
- **Correlation checks prevent redundancy**: Strong correlations between certain spend and interaction metrics guided feature pruning.

## Current Status & Next Steps
The current phase has delivered a clean, balanced, and ethically sound feature set ready for modelling. The next stage will involve:
- Building and comparing classification models.
- Addressing class imbalance through resampling or algorithmic weighting.
- Evaluating models using precision, recall, F1-score, and ROC-AUC.
The end goal is a robust, interpretable model that supports targeted, data-driven customer retention strategies.
#
**Author**: Kush S.

**Programme**: Lloyds Bank - Forage Virtual Experience

**Focus Areas**: Predictive Analytics, Customer Churn Modelling, Ethical Feature Engineering
#
