# Coronavirus Pandemic Playbook

## Census Data
AGE AND SEX: https://data.census.gov/cedsci/table?g=0100000US,.04000.001&y=2019&tid=ACSST1Y2019.S0101&hidePreview=true&moe=false&tp=true

RACE: https://data.census.gov/cedsci/table?q=United%20States&t=Race%20and%20Ethnicity&g=0100000US.04000.001&y=2019&tid=ACSDT1Y2019.B02001&tp=true&hidePreview=true&moe=false


MD AGE AND SEX: https://data.census.gov/cedsci/table?q=maryland%20age%20and%20sex&y=2019&tid=ACSST1Y2019.S0101&hidePreview=true&tp=true

MD RACE: https://data.census.gov/cedsci/table?q=maryland%20race&y=2019&tid=ACSDT1Y2019.B02001&hidePreview=true&tp=false&moe=false

## Machine Learning

### Preliminary Data Spitting / Testing Sets

We used Maryland COVID-19 data as our preliminary data, which is a subset of our main data table. We chose one state to work with first as the size of the data is significantly smaller. This helped the model run faster for the testing phase. As the model worked with the Maryland data, we are assuming that the model will work for the whole dataset. For the preliminary data, we used features from age, gender, and race, hospitalization, ICU, and underlying conditions. This was most efficient for the preliminary model because these features were already included in the main dataset. 

### Preliminary Engineering and Feature Selection

The model we chose to use is a **supervised random forest model.** We chose supervised machine learning because we have labeled data (our features in tabular form) and outputs (whether someone has COVID-19 or not). The input data, or our features, has a paired outcome which is plugged in to train the model to predict outcomes. Supervised machine learning models have target variables, or variables, about which we want to gain a deeper understanding. In our case our target variable is whether or not a person has COVID-19. 

### Model Choice

We chose a random forest algorithm because it can handle many input variables of which we will have many. It can also account for null values, which we found many in our base dataset. The algorithm can run efficiently on large datasets (the original dataset before transformation had 27 million rows), and most importantly, random forest models can be used to rank the importance of input variables. This fits the question we are trying to answer perfectly - **what are the top factors that influence the spread of COVID?** A random forest model will help us rank the most influential factors. While a large number of trees in a random forest algorithm can be slow requiring a lot of computational power and resources, the advantages outweigh the disadvantages.

### Code for Random Forest Model

To create the random forest model, we first initialize the dependencies, notably the 'from sklearn.ensemble import RandomForestClassifier'

```
import pandas as pd
from path import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
```

After loading in the data, we process the data by defining the features or independent variables, and the target variables or dependent variables.

```
file_path = ("maryland_data.csv")
covid_df = pd.read_csv(file_path)

X = pd.get_dummies(covid_df, columns=["age", "gender", "race", "hospitalization", "ICU", "underlying conditions"]
X = X.drop(columns="covid cases", axis=1)

y = covid_df["covid cases"]
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
rf_model = RandomForestClassifier(n_estimators=128, random_state=1) 

rf_model = rf_model.fit(X_train_scaled, y_train)
```

We make predictions and then evaluate how well the model classified the data.

```
predictions = rf_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, predictions)

cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
cm_df

acc_score = accuracy_score(y_test, predictions)

print("Confusion Matrix")
display(cm_df)
print(f"Accuracy Score : {acc_score}")
print("Classification Report")
print(classification_report(y_test, predictions))
```

We finally rank the importance of the features and see which have the most impact on the output.

```
importances = rf_model.feature_importances_
importances

sorted(zip(rf_model.feature_importances_, X.columns), reverse=True)
```
