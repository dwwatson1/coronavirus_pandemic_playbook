# Coronavirus Pandemic Playbook

## Census Data
AGE AND SEX: https://data.census.gov/cedsci/table?g=0100000US,.04000.001&y=2019&tid=ACSST1Y2019.S0101&hidePreview=true&moe=false&tp=true

RACE: https://data.census.gov/cedsci/table?q=United%20States&t=Race%20and%20Ethnicity&g=0100000US.04000.001&y=2019&tid=ACSDT1Y2019.B02001&tp=true&hidePreview=true&moe=false


MD AGE AND SEX: https://data.census.gov/cedsci/table?q=maryland%20age%20and%20sex&y=2019&tid=ACSST1Y2019.S0101&hidePreview=true&tp=true

MD RACE: https://data.census.gov/cedsci/table?q=maryland%20race&y=2019&tid=ACSDT1Y2019.B02001&hidePreview=true&tp=false&moe=false

## ETL Method

### Extracting the Data
Our main dataset was COVID-19 CDC data, which consists of unique patient information spanning 19 columns and 27 million rows. With such a large dataset, we used Amazon S3 to store the data and used Google Colab with Pyspark to access and load the data.

### Transforming the Data
With the data loaded, we could now transform our data. We first filtered the data to be between March 2020 and December 2020. We chose this date range because March 2020 was when the United States declared COVID-19 a pandemic and December 2020 was when the first vaccine was administered in the United States. After filtering the data by date, we dropped many columns from the dataset for either or both of these reasons: 1) there were too many missing values for the variable to be usable and/or 2) the variable was not useful for our analysis. After dropping the unnecessary columns, the data was left with four variables: res_state, age_group, sex, and race. The dataset also had missing values which were identified in the data as either "Missing", "Unknown", or "NA". We replaced all the "Missing" and "Unknown" values to be "NA" for simplicity in identifying the missing values. The data was then exported to a CSV file where it was then imported into SQL for storage and further querying.

### Loading the Data
We used SQL to store the data and query it so that the data would be organized by state with the values becoming our new features. For example, we now have "Male" and "Female" as features of our data with totals of each for each state, whereas in the base CDC data, "sex" was the feature and "Male" and "Female" were values for the unique patients. The other features we are using were imported from their respective CSV files and joined to this main table. Using the U.S. Census data, we were able to create the features for those who do not have COVID by subtracting the number of people with COVID by the total numbers for each state. For example, to find the total number of females who do not have COVID for the state of Maryland, we subtracted the total number of females with COVID from the total population of the state of Maryland.

### Handling Missing Values
Currently, missing values are their own features in our dataset where we have them for age, sex, and gender, which could potentially result in poor performance of the machine learning model. We have a couple other potential ideas on how we could handle the missing values:

- As the features with missing values are categorical variables, we could impute the missing values by using the mode.
- We can predict the missing values for the categorical variables by using a classification model. We would split the data as such:
  - y_train: rows from data with non null values
  - y_test: rows from data with null values
  - X_train: Dataset except data features with non null values
  - X_test: Dataset except data features with null values

## Machine Learning

### Model Choice
The model we chose to use is a **supervised random forest regression model.** We chose supervised machine learning because we have labeled data (our features in tabular form) and outputs. The input data, or our features, has a paired outcome which is plugged in to train the model to predict outcomes. Supervised machine learning models have target variables, or dependent variables, about which we want to gain a deeper understanding. In our case our target variable is how much effect COVID had on a state's population by looking at the total number of COVID cases divided by the total state population.

We chose a random forest algorithm because it can handle many input variables of which we have many. It can also account for null values, which we found many in our base dataset. The algorithm can run efficiently on large datasets, and most importantly, random forest models can be used to rank the importance of input variables. This fits the question we are trying to answer perfectly - **what are the top factors that influence the spread of COVID?** A random forest model will help us rank the most influential factors. Since we have a large dataset with many features, as well as both continuous and categorical non-linear variables, a random forest algorithm will be more efficient and more accurate than a simple linear regression. While a large number of trees in a random forest algorithm can be slow requiring a lot of computational power and resources, the advantages outweigh the disadvantages.

### Code for Random Forest Model
To create the random forest model, we first initialize the dependencies, notably the 'from sklearn.ensemble import RandomForestRegressor'.

```
import pandas as pd
from path import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```

After loading in the data, we use one hot encoding to account for null values and convert categorical variables to integer data.

```
file_path = ("COVID_MARCH2020_DEC2020_TOTALS_PROJECT4.csv")
covid_df = pd.read_csv(file_path)

covid_cat = covid_df.dtypes[covid_df.dtypes == "object"].index.tolist()

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)
encode_df = pd.DataFrame(enc.fit_transform(covid_df[covid_cat]))

encode_df.columns = enc.get_feature_names(covid_cat)
encode_df.head()
```

We then merge the one hot encoded features to the main dataframe and drop the originals.

```
covid_df = covid_df.merge(encode_df, left_index=True, right_index=True)
covid_df = covid_df.drop(covid_cat,1)
covid_df
```

We split our preprocessed data into our features and target variables.

```
y = covid_df["case_pop"].ravel()
X = covid_df.copy()
X = X.drop("case_pop", axis=1)
```

We then split the data into training and testing sets and scale the data. We set random_state to a number in the testing phase so that we can consistently see the same results when the test model is run (this could possibly be removed for the final model).

```
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

scaler = StandardScaler()

X_scaler = scaler.fit(X_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

We initialize the random forest classifier and fit the model. We set n_estimators to 128 because best practice is to use between 64 and 128 forests. Generally, the higher the number, the stronger and more stable the predictions are. Given that this is a test model, it is reasonable to assume the model might be able to handle 128 forests.

```
rf_model = RandomForestRegressor(n_estimators=128, random_state=1) 

rf_model = rf_model.fit(X_train_scaled, y_train)
```

We make predictions and then evaluate how well the model classified the data.

```
predictions = rf_model.predict(X_test_scaled)

acc_score = accuracy_score(y_test, predictions)

print(f"Accuracy Score : {acc_score}")
```

We finally rank the importance of the features and see which have the most impact on the output.

```
importances = rf_model.feature_importances_
importances

sorted(zip(rf_model.feature_importances_, X.columns), reverse=True)
```

