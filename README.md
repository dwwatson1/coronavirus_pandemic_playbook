# Coronavirus Pandemic Playbook

## Project Topic, Background, Audience

The term 'pandemic playbook' circulated in the news during the beginning of the COVID-19 pandemic. U.S. President Barack Obama's team had outlined how to respond to infectious diseases and biological incidents so future administrations would be prepared to respond to the next pandemic or biological threat. While the federal government prepared guidelines, state governments from the 50 states + DC were woefully unprepared for a pandemic. The COVID-19 pandemic brought the [deepest recession since the end of WWII](https://www.brookings.edu/research/social-and-economic-impact-of-covid-19/) as the global economy shrunk by 3.5% and [114 million](https://www.weforum.org/agenda/2021/02/covid-employment-global-job-loss/) people lost their jobs in 2020. The impact of this shock is likely to be felt for years to come.

The National Governors Association, a nonpartisan organizations comprised of governors in the U.S., tasked with creating an updated playbook for state governments. They have asked us to provide a comprehensive review of factors that led to the spread of COVID-19 cases in states across the United States. We will be presenting our at the next National Governors Association annual conference in late 2021.

### Project Goal
Drawing on data from CDC, U.S. Census, and many other sources, our goal is to determine which social, economic, and political factors contributed to the spread of COVID-19. There is evidence that if states were more prepared to handle a pandemic, economic performance would not have suffered as it did in 2020. Our nation's governors have the opportunity to learn where our state's weak points were that led to these incredible economic losses and mitigate them in a future pandemic. Our team is confident that our machine learning algorithm will predict which factors contributed the most to the spread of respiratory diseases like COVID-19. The information will valuable for state lawmakers' future economic and social political decisions.

### Project Factors 

Given our audience for the project, the data we've obtained for each factor will be organized by state. 

**Target Variable**
* Number of COVID-19 Cases / State Population

**Social Factors**
* Sex
* Age
* Race

**Geographical Factors**
* Population Density
* Commercial Airports

**Economic Factors**
* Median Household Income

**Political Factors**
* State Mandates / COVID-19 rules
* Political Leaning

### Questions to Investigate During Project
1. Which social, economic, geographical, or political factors contributed the most the spread of the disease?
2. Which category of factor contributed the most the spread of the disease?
3. Is there a connection between state policy or political leaning (i.e. mask mandate) and the spread of COVID-19 within the state
4. Do we need to account for the size of the population that didn't have COVID-19 when using a machine learning model?

## Data Exploration and Analysis Phases

### Data Exploration Overview

We began the project by looking at the entirety of COVID-19 CDC data, which consists of 27 million rows and 19 columns of unique patient information. We quickly realized that if we wanted to replicate the spread of COVID-19 based on any factor, we needed to account for the population that didn't have the disease. We established **Number of COVID-19 Cases / State Population** as our target variable. We found ratio would be easier to handle data-wise than working with large population datasets or creating pseudo population data. Next, we moved on to categorical factors. 

For social factors, we looked at U.S. Census data estimates for information on sex, age, and race. We observed that both datasets had either state abbreviations or states spelled out with their full names. We knew we could join data tables by state, so we focused our efforts on finding geographical, economic, and political factors with state columns already available.

### Data Analysis Overview

Once we have joined all the factor tables in SQL, we will run 

### Datasets and Sources

* [COVID-19 Cases by Age, Sex, Race](https://github.com/dwwatson1/coronavirus_pandemic_playbook/blob/main/Resources/COVID_MARCH2020_DEC2020_TOTALS_PROJECT4.csv) Source: U.S. Census and CDC
* [U.S. Commercial Airports by State](https://github.com/dwwatson1/coronavirus_pandemic_playbook/blob/main/Resources/Group4%20Airport%20By%20Area.csv) Source: FAA
* [State Mask Mandate Policy/Political Affiliation by State](https://github.com/dwwatson1/coronavirus_pandemic_playbook/blob/main/Resources/state_factors_from_gallup.csv) Source: Gallup
* [Median Household Income by State](https://github.com/dwwatson1/coronavirus_pandemic_playbook/blob/main/Resources/household_income_by_state.csv) Source: U.S. Census

### Description of Data Sources

Our primary dataset for this project consist of over 27 million rows of unique patient Covid-19 data and was sourced from the Center for Disease Control and Prevention (CDC) Case Surveillance Public Use Data.  It consists of 19 columns of patient specific attributes that will be reduced to 16 columns. Two other tables have been identified, and include data based on Religion by State and State Covid-19 Policy Mandates by state. These tables will be joined to the reconstructed primary database which will be indexed by the 50 US states and its territories. As we refine our Questions to Investigate, we may see fit to remove more columns due to numerous missing values. 

Applying the SQL code on the primary dataset, the reconstructed dataframe will be grouped by state and the data values will become the columns with the patients become the measure to be counted.  In brief, the current primary dataset will be grouped with columns expanded.

## Database

### Database Schema ERD

![CORONAVIRUS_PANDEMIC_PLAYBOOK_wCENSUSdata_ERD](https://github.com/dwwatson1/coronavirus_pandemic_playbook/blob/main/Images/CORONAVIRUS_PANDEMIC_PLAYBOOK_wCENSUSdata_ERD.png)

### Building the Database

[Database Storing Overview](https://github.com/dwwatson1/coronavirus_pandemic_playbook/blob/main/Resources/Project%204%20Database%20SQL.txt)

**Steps**

1. We chose input data from CDC from March - December 2020 because March marked when the U.S. declared a state of emergency and December was when the first COVID-19 vaccine dose was administered and U.S. Census data. After storing the data in pgAdmin - PostgreSQL, we selected age group, state, sex, and race, columns from both datasets.
2. We then created a table to hold the input data from CDC and U.S. Census. We called the table CDC_INPUTDB_CLEANED
3. We identified the age group options available in the data and created an age group table
4. We inserted the age group table into the CDC_INPUTDB_CLEANED table
5. We repeated steps 3 and 4 for sex and race factors
6. We read the query to summarize counts and selected Maryland as our test state for our new table
7. Before creating the final table that include counts for the segments of each factor, we summarized each column's datatype.
8. We created our final table to hold the total value counts: COVID_MARCH2020_DEC2020_PROJECT4
9. Then we added up the counts for the segments of each factor and inserted into the COVID_MARCH2020_DEC2020_PROJECT4 table
10. Exported to [COVID-19 Cases by Age, Sex, Race](https://github.com/dwwatson1/coronavirus_pandemic_playbook/blob/main/Resources/COVID_MARCH2020_DEC2020_TOTALS_PROJECT4.csv)

### Data Dictionary

**PLACE GROUP 4 DATA DICT .PNG IMAGE HERE, DAVID**

### Database ETL Method

Our main data table has "Missing", "Unknown", and "NA" values. Because these values are similar, we replaced these values to be all NA. In order for our machine learning model to process the data, we replaced all the NA values with 0. By replacing the NA's with 0, we will see that there are fewer values in certain columns that do not add up to the total number of COVID cases. For example, since there were some missing values for whether the COVID case person was either Male or Female, the total Male/Female columns will not add up to the total cases. To account for this, we will use the SMOTE oversampling technique.

## Machine Learning

### Preliminary Data Splitting / Testing Sets
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

## Dashboard

### Blueprint and Interactive Elements

One of the interactive elements we are using is the filter action. Filter actions send information between worksheets. Typically, a filter action sends information from a selected mark to another sheet showing related information. Behind the scenes, filter actions send data values from the relevant source fields as filters to the target sheet and dashboards.

For example, in a view showing the states, when a user selects a particular state name, a filter action can show all state values for all the displayed variables. 
User can select marks to see information about a specific data filed. One can also select an individual mark or multiple ones by holding down the Ctrl key (for Windows) or the Command key (macOS).

When you select marks in the view, all other marks are dimmed to draw attention to the selection. The selection is saved with the workbook. Quick data view can also be done by one of the run-on options; hovering your mouse on the charts/marks. 

We have also created a simple HTML file to show the dashboard in a dedicated webpage with another interactive element where users can download the analysis into PDF file. 
 

### [Tableau Dashboard Demo](https://public.tableau.com/views/ALLSTATESDATAMARCHtoDEC2020/Dashboard4?:language=en-US&:display_count=n&:origin=viz_share_link)

## Appendix 

### Roles

* **Project Manager**
    * David
* **Database Storage**
    * Dinah
    * Kimi
    * Michael
* **Data Cleaning and Analysis**
    * Dinah
    * Kimi
    * Michael
* **Machine Learning Model**
    * Michael
* **Presentation of Findings**
    * Yodit (Tableau)
    * Jack (Tableau
    * David (approver) & Team (GitHub)

### Technologies Used

* **Database Storage**
    * pgAdmin - PostgreSQL
    * AWS RDS
* **Data Cleaning and Analysis**
    * Juypter Notebook - Pandas
* **Machine Learning Model**
    * Google Collab Notebook
* **Presentation of Findings**
    * Tableau Public
    * GitHub

### Communication Protocol 

* [Project Checklist](https://docs.google.com/spreadsheets/d/1G9lvPyMrlkjnYT-qGigKpNdVk72A9Zu0Je7hyy8Q6ug/edit?usp=sharing)
* [Group meeting agendas](https://drive.google.com/drive/folders/1sMOLvKQO-S99917fQL9axuocZujgKNZQ?usp=sharing)

We are meeting twice a week outside of class on Zoom and consistently communicating over Slack. David has established best pratices in GitHub, so we don't overwrite each other's work.
