# Coronavirus Pandemic Playbook

## Project Topic, Background, Audience

The term 'pandemic playbook' circulated in the news during the beginning of the COVID-19 pandemic. U.S. President Barack Obama's team had outlined how to respond to infectious diseases and biological incidents so future administrations would be prepared to respond to the next pandemic or biological threat. While the federal government prepared guidelines, state governments from the 50 states + DC were woefully unprepared for a pandemic. The COVID-19 pandemic brought the [deepest recession since the end of WWII](https://www.brookings.edu/research/social-and-economic-impact-of-covid-19/) as the global economy shrunk by 3.5% and [114 million](https://www.weforum.org/agenda/2021/02/covid-employment-global-job-loss/) people lost their jobs in 2020. The impact of this shock is likely to be felt for years to come.

The National Governors Association, a nonpartisan organizations comprised of governors in the U.S., tasked with creating an updated playbook for state governments. They have asked us to provide a comprehensive review of factors that led to the spread of COVID-19 cases in states across the United States. We will be presenting our at the next National Governors Association annual conference in late 2021.

### Project Goal
Drawing on data from CDC, U.S. Census, and many other sources, our goal is to determine which social, economic, and policy factors contributed to the spread of COVID-19. There is evidence that if states were more prepared to handle a pandemic, economic performance would not have suffered as it did in 2020. Our nation's governors have the opportunity to learn where our state's weak points were that led to these incredible economic losses and mitigate them in a future pandemic. Our team is confident that our machine learning algorithm will predict which factors contributed the most to the spread of respiratory diseases like COVID-19. The information will valuable for state lawmakers' future economic and social policy decisions.

### Project Factors 

Given our audience for the project, the data we've obtained for each factor will be organized by state. 

* Number of COVID-19 Cases **(Target Variable)**
* Gender
* Age
* Race
* State Mandates / COVID-19 rules
* Median Household Income
* Population Density
* Commercial Airports

### Questions to Investigate During Project
1. What is the population per state at the beginning and end of the pandemic?
2. How has race played a role in the spread of the COVID-19 pandemic?
3. Could the level of poverty and inequality affect the spread of COVID-19? If so what is the impact?
4. What influence did State Covid Mandate policies have on testing, hospital utilization, ICU admittance, and death outcome.?
5. What were the top 5 US States & territories impacted by Covid-19? 
6. During periods of Covid-19 case spikes, were there geographical or state areas that trended with these spikes?
7. Did religious affiliation have an influence on the number of Covid cases prior to vaccine distribution?

## Data Exploration and Analysis Phases

### Data Exploration Overview

### Data Analysis Overview

## Database

### Description of Data Sources

Our primary dataset for this project consist of over 27 million rows of unique patient Covid-19 data and was sourced from the Center for Disease Control and Prevention (CDC) Case Surveillance Public Use Data.  It consists of 19 columns of patient specific attributes that will be reduced to 16 columns. Two other tables have been identified, and include data based on Religion by State and State Covid-19 Policy Mandates by state. These tables will be joined to the reconstructed primary database which will be indexed by the 50 US states and its territories. As we refine our Questions to Investigate, we may see fit to remove more columns due to numerous missing values. 

Applying the SQL code on the primary dataset, the reconstructed dataframe will be grouped by state and the data values will become the columns with the patients become the measure to be counted.  In brief, the current primary dataset will be grouped with columns expanded.

Maryland State sub dataset:
![image](https://user-images.githubusercontent.com/79073778/125868128-740d4848-1f0b-4f32-9f35-b0e6a3a60846.png)

Testing our data strategy, a subset of the primary dataset has been created for one state. That one state (MD) data subset consist of 225,815 rows to which the SQL code has been applied as a preparation step for reconstructing the entire database.  The following is sample work:

![image](https://user-images.githubusercontent.com/79073778/125867840-4ed53e91-338b-47ea-b311-6a82f029610e.png)

![image](https://user-images.githubusercontent.com/79073778/125867942-26a4b027-bd93-44dd-bf8a-e0e9ee6965f7.png)

![image](https://user-images.githubusercontent.com/79073778/125868019-79dd73d9-244c-4d03-8502-58649666cf8f.png)

![image](https://user-images.githubusercontent.com/79073778/125867494-eae850e9-66a7-46db-b40b-efb4f1b54959.png)

![image](https://user-images.githubusercontent.com/79073778/125867771-34edf60e-f0b9-444c-8d6e-f6afde805352.png)

### Data Dictionary

| Column Name  | Description | Type | 
| ------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------- |
| case_month  | The earlier of month the Clinical Date (date related to the illness or specimen collection) or the Date Received by CDC  | DATE  |
| res_state | State of residence  | TEXT  |
| state_fips_code  | Numeric two-digit code to identify states  | TEXT  |
| res_county  | County of residence  | TEXT  |
| county_fips_code  | Numeric five-digit codes to identify counties  | TEXT  |
| age_group  | 0 - 17 years; 18 - 49 years; 50 - 64 years; 65 + years; Unknown; Missing; NA, if value suppressed for privacy protection | TEXT  |
| sex  | Female; Male; Other; Unknown; Missing; NA, if value suppressed for privacy protection  | TEXT  |
| race  | American Indian/Alaska Native; Asian; Black; Multiple/Other; Native Hawaiian/Other Pacific Islander; White; Unknown; Missing; NA, if value suppressed for privacy)  | TEXT  |
| ethnicity  | Hispanic; Non-Hispanic; Unknown; Missing; NA, if value suppressed for privacy protection | TEXT  |
| exposure_yn  | In the 14 days prior to illness onset, did the patient have any of the following known exposures: domestic travel, international travel, cruise ship or vessel travel as a passenger or crew member, workplace, airport/airplane, adult congregate living facility, etc  | BOOLEAN  |
| current_status  | What is the current status of this person?  | BOOLEAN  |
| symptom_status  | What is the symptom status of this person?  | BOOLEAN  |
| hosp_yn  | Was the patient hospitalized?  | BOOLEAN  |
| icu_yn  | Was the patient admitted to an intensive care unit (ICU)?  | BOOLEAN  |
| death_yn  | Did the patient die as a result of this illness?  | BOOLEAN  |
| underlying_conditions_YN  | Did the patient have one or more of the underlying medical conditions and risk behaviors  | BOOLEAN  |

### SQL Database ERD Schema

![COVID_Pandemic_ERD](https://github.com/dwwatson1/coronavirus_pandemic_playbook/blob/main/Images/COVID_Pandemic_ERD.png)

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

We will create an HTML/CSS portfolio to showcase our project and Bootstrap components to polish and customize the portfolio. We will also use JavaScript functions to display dynamic and interactive dashboard. 

### [Tableau Dashboard Demo](https://public.tableau.com/views/ALLSTATESDATAMARCHtoDEC2020/Dashboard2?:language=en-US&:display_count=n&:origin=viz_share_link)

## Resources

### Data Sources List

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
