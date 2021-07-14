# Coronavirus Pandemic Playbook

## Project Topic and Description

### Topic

Investigating the six most important factors that led to the spread of COVID-19 cases in states across the United States.
* Gender
* Age
* Weight
* Race
* Political Party
* Religious Affiliation
* Income Level
* Population Density
* Number of Commercial Airports

### Audience

The National Governors Association has tasked our group, the **COVID-19 Rapid Response Group: Preparing for the Next Pandemic** will be presenting our pandemic playbook to at a COVID-19 response conference with the National Governors Association and to the United States Conference of Mayors. The two nonpartisan organizations comprised of governors from the U.S.'s 55 states and territories and mayors of U.S. cities with population of 30,000 or more. Typically these groups meet separately to share strategies across their jurisdictions, however, this conference is a rare opportunity for all executives to gather and learn how to better respond to the next pandemic and minimize the spread, deaths, and economic impact. 

### Why Should We Care?

As a capitalist society, some economic measures of success for the U.S. are jobs created and GDP growth. The COVID-19 brought the [deepest recession since the end of WWII](https://www.brookings.edu/research/social-and-economic-impact-of-covid-19/) as the global economy shrunk by 3.5% and [114 million](https://www.weforum.org/agenda/2021/02/covid-employment-global-job-loss/) people lost their jobs in 2020. The impact of this shock is likely to be felt for years to come.

[The Brookings Institute](https://www.brookings.edu/research/social-and-economic-impact-of-covid-19/) identified **state capacity** as one of three pre-existing conditions that amplified the impact of the shock. The COVID-19 crisis posed a critical challenge for policymakers as they needed to quickly reach workers and households during the abrupt economic crisis. There is evidence that if states were more prepared to handle a pandemic, economic performance would not have suffered as it did in 2020. Our nation's governors and mayors have the opportunity to learn where our countries weak points are that led to these incredible economic losses and mitigate them in a future pandemic. 

## Technologies Used

* **Database storage**
** We will use SQL database tool to extract, organize and retrieve our data.
* **Data Cleaning and Analysis**
** Pandas will be used to clean and transform the data and perform an exploratory analysis. 

## Communication Protocol 

[Group meeting agendas](https://drive.google.com/drive/folders/1sMOLvKQO-S99917fQL9axuocZujgKNZQ?usp=sharing)

We are utilizing the available and most suitable resources as our communication tools. Zoom and Slack. We are aiming to meet twice a week in addition to meeting and discussing over the regular virtual class hours. 

We have created a group and direct messages for group members in Slack, and we may use this channel for any cases of emergency. 

### Data Sources for Project
**Main Data Source**
* [Case Surveillance Public Use Data with Geography](https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data-with-Ge/n8mc-b4w4/data)
* [U.S. Census Data](https://www.census.gov/data/developers/data-sets.html)

**COVID-19 State Mandate Data**
* [US State Level Policy Tracker](https://github.com/govex/COVID-19/tree/govex_data/data_tables/policy_data/table_data/Current)
 
**Other Data Sources Used**

We used these other data sources to find columns that matched our schema.
* [John Hopkins Coronavirus Data](https://coronavirus.jhu.edu/data/new-cases-50-states)
* [US Dept of Health Data Sources by State](https://github.com/CSSEGISandData/COVID-19)
* [Additional data source that we are considering](https://docs.google.com/document/d/10i01u6oQAUVCbk5VTL6G0rIsTF9JlO1I90XTCDXWTCA/edit)
* [Another possible data APIs](https://blogs.mulesoft.com/dev-guides/track-covid-19/)
* [A database that we can consider](https://covidtracking.com/data/download)
* [APM Research Lab: The Color of Coronavirus cvs files](https://www.apmresearchlab.org/covid/deaths-by-race)
* [COVID-19 original and derived datasets (JHU, NY Times, ECDC)](https://github.com/cipriancraciun/covid19-datasets)

### Questions to Investigate During Project
1. What is the population per state at the beginning and end of the pandemic?
2. How has race played a role in the spread of the COVID-19 pandemic?
3. Could the level of poverty and inequality affect the spread of COVID-19? If so what is the impact?
4. Did having medical insurance play a role in the cure and deaths?
5. What were the top 5 MSAs (Metropolitan Statistical Areas) impacted by Covid-19? (Def: MSA is a geographical region with a relatively high population density at its core and close economic ties throughout the area.)
6. What were the top 20 uSAs (Micropolitan Statistical Areas) impacted by Covid-19?
7. During periods of Covid-19 case spikes, were there geographical or state areas that trended with these spikes?
8. Did political affiliation of areas have an influence on the number of Covid cases prior to vaccine distribution?

## DATABASE

### Data Dictionary

![image](https://user-images.githubusercontent.com/79073778/125699214-0a67f9a7-cd0d-4933-b7a7-8c7f62741baf.png)

### SQL Database Schema

#### States

State_Init VARCHAR PK

Pop_Density INTEGER

Income INTEGER

Race VARCHAR

Zipcode VARCHAR

#### Covid_Cases

Case_Totals INTEGER PK
State_Init VARCHAR FK >- States.State_Init
Death_Totals INTEGER 
Hospital_Utilz INTEGER

#### Demographics

Case_Totals INTEGER PK
State_Init VARCHAR FK >- States.State_Init
Age INTEGER
Politics VARCHAR 
Religion VARCHAR
Gender VARCHAR

#### Identity

Race VARCHAR PK
State_Init VARCHAR FK >- States.State_Init
Case_Totals INTEGER FK >- Covid_Cases.Case_Totals
Death_Totals INTEGER FK >- Covid_Cases.Case_Totals
Income INTEGER
Health_Issues VARCHAR 
 
### Method

#### Extract, Transform, and Load the Data

Our main data table has "Missing", "Unknown", and "NA" values. Because these values are similar, we replaced these values to be all NA. In order for our machine learning model to process the data, we replaced all the NA values with 0. By replacing the NA's with 0, we will see that there are fewer values in certain columns that do not add up to the total number of COVID cases. For example, since there were some missing values for whether the COVID case person was either Male or Female, the total Male/Female columns will not add up to the total cases. To account for this, we will use the SMOTE oversampling technique.

#### Data Dictionary

| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |

## Machine Learning

### Preliminary Data Spitting / Testing Sets
We used Maryland COVID-19 data as our preliminary data, which is a subset of our main data table. We chose one state to work with first as the size of the data is significantly smaller. This helped the model run faster for the testing phase. As the model worked with the Maryland data, we are assuming that the model will work for the whole dataset. For the preliminary data, we used features from age, gender, and race, hospitalization, ICU, and underlying conditions. This was most efficient for the preliminary model because these features were already included in the main dataset. 

### Preliminary Engineering and Feature Selection
The model we chose to use is a **supervised random forest model.** We chose supervised machine learning because we have labeled data (our features in tabular form) and outputs (whether someone has COVID-19 or not). The input data, or our features, has a paired outcome which is plugged in to train the model to predict outcomes. Supervised machine learning models have target variables, or dependent variables, about which we want to gain a deeper understanding. In our case our target variable is whether or not a person has COVID-19. 

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

#### Dashboard

We will create an HTML/CSS portfolio to showcase our project and Bootstrap components to polish and customize the portfolio. We will also use JavaScript functions to display dynamic and interactive dashboard. 

#### Machine Learning 

We will use Scikit-learn machine learning library for Python to reshape, stratify, split, test, train …. our data. 
