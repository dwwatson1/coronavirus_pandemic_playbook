# Module 20

# Coronavirus Pandemic Playbook
## Group 4
	Leggett, Michael
	Essilfie-Bondzie, DINAH
	Teamir rezene, Yodit
	Ellerbe, Kimberly
	Gross, Jack
	Watson, David

Checklist
-----------
		8/38 completed  			
					
	✓	Rubric Task	Comments	Status	Date Completed
Presentation	FALSE	Presentation - Content			
	TRUE	Selected topic			
	TRUE	Stated reason why the topic was selected			
	FALSE	Description of the source data			
	FALSE	Questions we hope to answer			
	FALSE	Description of the data exploration phase			
	FALSE	Description of the analysis phase			
	FALSE	Presentation - Format TBD	We aren't using Google Slides. Need an alternative. GitHub Readme?		
GitHub	FALSE	GIthub - Master Branch			
	FALSE	All code in the master branch is production-ready			
	FALSE	Some code necessary to complete the machine learning portion of the project			
	FALSE	Github - README			
	FALSE	Includes description of the communication protocols	Nearly completed.		
	FALSE	Includes outline of the project 	may include images, but should be easy to follow		
	FALSE	All other components of project in README as well	Create section for technologies used		
	FALSE	GIthub - Individual Branches			
	FALSE	Have at least one for each team member			
	FALSE	Each team member meets the GitHub commit minimum	At least 4 commits for the duration of the second segment and 8 commits in total. Nick won't be a stickler about this but lets make sure we're all contributing		
Machine Learning Model	TRUE	Machine Learning Model			
	TRUE	Code for the machine learning model			
	TRUE	Description of preliminary feature data preprocessing			
	TRUE	Description of preliminary feature engineering and preliminary feature selection, including decision-making process			
	TRUE	Description of how data was split into training and testing sets			
	TRUE	Explain model choice, including limitations, and benefits			
Database	FALSE	Database			
	FALSE	Stores static data for use during the project			
	FALSE	Interfaces with the project in some format (e.g. scrapting updates in database or database connects to the model)			
	FALSE	Includes at least 2 tables 			
	FALSE	Includes at least one join using the database language (not including any joins in Pandas)			
	FALSE	Includes at least one connection string (using SQLAlchemy or PyMongo)			
	FALSE	Provide ERD with relationships	Only if using SQL database, so yes, we'll need this		
Dashboard	FALSE	Dashboard			
	FALSE	A blueprint for the dashboard is created and includes:			
	FALSE	Storyboard in Tableau	Not in Google Slides		
	FALSE	Description of the tool that will be used to create final dashboard	README file		
	FALSE	Description of interactive elements			
Extra Items Needed 	FALSE	List of technologies used in README			
	FALSE	Description of how we will use Tableau in README			
July12					
					
----------
# Technologies Used

## Machine Learning
We will use Scikit-learn machine learning library for Python to reshape, stratify, split, test, train …. our data.

## Data Cleaning and Analysis
Initial table cleaning is done with Pandas in Colab
Where the fields with the required data is picked and all others dropped.
-----------
# Extract from file 1
# Step 1
application_df = pd.read_csv("Provisional_COVID-19_Deaths_by_Sex_and_Age.csv")
application_df.head()
# Determine the number of unique values in each column
# Step 2
application_df.nunique()
Data As Of                                     1
Start Date                                    19
End Date                                      19
Group                                          3
Year                                           2
Month                                         12
State                                         54
Sex                                            3
Age Group                                     17
COVID-19 Deaths                             2533
Total Deaths                                6672
Pneumonia Deaths                            2396
Pneumonia and COVID-19 Deaths               1836
Influenza Deaths                             325
Pneumonia, Influenza, or COVID-19 Deaths    2963
Footnote                                       1
# Step 3
application_df.shape
(60588, 16)
# Step 4
new_Provisional_COVID_Deaths_by_Sex_and_Age_df=application_df[["Data As Of","Start Date","End Date","Group","State", "Sex", "Age Group"]]
# Step 5
new_Provisional_COVID_Deaths_by_Sex_and_Age_df.isnull().sum()
# Determine the number of unique values in each column
# Step 6
new_Provisional_COVID_Deaths_by_Sex_and_Age_df.nunique()
# Step 7
df_Covid_Sex_Age=new_Provisional_COVID_Deaths_by_Sex_and_Age_df.rename(columns={'Data As Of': 'AsOfDate','Start Date':'StartDate','End Date':'EndDate','Group' : 'Grouping','Age Group':'AgeGroup'}, inplace=False)
df_Covid_Sex_Age.head
[60588 rows x 7 columns]>
# Step 8
df_Covid_Sex_Age.nunique()
AsOfDate      1
StartDate    19
EndDate      19
Grouping      3
State        54
Sex           3
AgeGroup     17
# Save and export your results to an csv file - Provisional_COVID-19_Deaths_by_Sex_and_Age
# File 1
# Step 9
df_Covid_Sex_Age.to_csv("COVID_Deaths_by_Sex_and_Age.csv")


# Extract from file 2
# Step 1
app_df = pd.read_csv("United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv")
app_df.head()
# Determine the number of unique values in each column
# Step 2
app_df.nunique()
submission_date      531
state                 60
tot_cases          23510
conf_cases         13043
prob_cases          9188
new_case            5133
pnew_case           1757
tot_death          10677
conf_death          7822
prob_death          2291
new_death            444
pnew_death           252
created_at          1755
consent_cases          2
consent_deaths         2
# Step 3
app_df.shape
(31860, 15)
# Step 4
new_US_COVID_Cases_and_Deaths_by_State_over_Time_df=app_df[["submission_date","state","tot_cases","new_case","tot_death", "new_death", "created_at"]]
# Step 5
new_US_COVID_Cases_and_Deaths_by_State_over_Time_df.isnull().sum()
# Determine the number of unique values in each column
# Step 6
new_US_COVID_Cases_and_Deaths_by_State_over_Time_df.nunique()
# Step 7
df_COVID_Cases_Deaths_by_State=new_US_COVID_Cases_and_Deaths_by_State_over_Time_df.rename(columns={'submission_date': 'AsOfDate'}, inplace=False)
df_COVID_Cases_Deaths_by_State.head
[31860 rows x 7 columns]>
# Step 8
df_COVID_Cases_Deaths_by_State.nunique()
AsOfDate        531
state            60
tot_cases     23510
new_case       5133
tot_death     10677
new_death       444
created_at     1755
# Save and export your results to an csv file - United_States_COVID-19_Cases_and_Deaths_by_State_over_Time
# File 2
# Step 9
df_COVID_Cases_Deaths_by_State.to_csv("COVID_Cases_Deaths_by_State.csv")


# Extract from file 3
# Step 1
df_application = pd.read_csv("Conditions_Contributing_to_COVID-19_Deaths__by_State_and_Age.csv")
df_application.head()
# Determine the number of unique values in each column
# Step 2
df_application.nunique()
Data As Of               1
Start Date              19
End Date                19
Group                    3
Year                     2
Month                   12
State                   54
Condition Group         12
Condition               23
ICD10_codes             23
Age Group               10
COVID-19 Deaths       3426
Number of Mentions    3561
Flag                     1
# Step 3
df_application.shape
(273240, 14)
# Step 4
new_Conditions_Contributing_COVID_Deaths_by_State_and_Age_df=df_application[["Data As Of","Start Date", "End Date","Group", "State", "Condition", "Condition Group", "Age Group","COVID-19 Deaths"]]
# Step 5
new_Conditions_Contributing_COVID_Deaths_by_State_and_Age_df.isnull().sum()
# Determine the number of unique values in each column
# Step 7
new_Conditions_Contributing_COVID_Deaths_by_State_and_Age_df.nunique()
# Step 7
df_Covid_Contributing_Deaths=new_Conditions_Contributing_COVID_Deaths_by_State_and_Age_df.rename(columns={'Data As Of': 'AsOfDate','Start Date':'StartDate','End Date':'EndDate', 'Group': 'Grouping','Age Group': 'AgeGroup','Condition Group':'ConditionGroup','COVID-19 Deaths' : 'Deaths'}, inplace=False)
df_Covid_Contributing_Deaths.head
[273240 rows x 9 columns]>
# Step 8
df_Covid_Contributing_Deaths.nunique()
AsOfDate             1
StartDate           19
EndDate             19
Grouping             3
State               54
Condition           23
ConditionGroup      12
AgeGroup            10
Deaths            3426
# Save and export your results to an csv file - Conditions_Contributing_to_COVID-19_Deaths__by_State_and_Age
# File 3
# Step 9
df_Covid_Contributing_Deaths.to_csv("COVID_Conditions_Contributing_Deaths.csv")


# Extract from file 4
# Step 1
df_app = pd.read_csv("cdc_database_cleaned.csv")
df_app.head()

# Determine the number of unique values in each column
# Step 2
df_app.nunique()
Unnamed: 0                  141944
case_month                      10
res_state                       41
state_fips_code                 30
res_county                     393
county_fips_code               497
age_group                        4
sex                              3
race                             6
ethnicity                        2
exposure_yn                      1
current_status                   2
symptom_status                   3
hosp_yn                          2
icu_yn                           2
death_yn                         2
underlying_conditions_yn         2

# Step 3
df_app.shape
(141944, 17)

# Step 4
new_cdc_database_df=df_app.drop(columns=["state_fips_code","res_county","county_fips_code"], axis=1)

# Step 5
new_cdc_database_df.isnull().sum()

# Determine the number of unique values in each column
# Step 6
new_cdc_database_df.nunique()

# Step 7
df_COVID_cdc_db=new_cdc_database_df.rename(columns={'Data As Of': 'AsOfDate','Start Date':'StartDate','End Date':'EndDate', 'Group': 'Grouping','Age Group': 'AgeGroup','Condition Group':'ConditionGroup','COVID-19 Deaths' : 'Deaths'}, inplace=False)
df_COVID_cdc_db.head
[141944 rows x 14 columns]>

# Step 8
df_COVID_cdc_db.nunique()
case_month                      10
res_state                       41
age_group                        4
sex                              3
race                             6
ethnicity                        2
exposure_yn                      1
current_status                   2
symptom_status                   3
hosp_yn                          2
icu_yn                           2
death_yn                         2
underlying_conditions_yn         2

# Save and export your results to an csv file - Conditions_Contributing_to_COVID-19_Deaths__by_State_and_Age
# File 4
# Step 9
df_COVID_cdc_db.to_csv("COVID_CDC_DATA.csv")

-----------


## Database Storage
Mongo is the database we intend to use, and we will integrate Flask to display the data.
We will use SQL database tool to extract, organize and retrieve our data.


Futher cleaning is done in PostgreSQL

In PGAdmin the following were imported into postgreSQL
COVID_Conditions_Contributing_Deaths.csv into covidcontributiingdeath
COVID_Cases_Deaths_by_State.csv into coviddeathsbystates
COVID_Deaths_by_Sex_and_Age.csv into coviddeathsbyagesex

SELECT * FROM COVIDCONTRIBUTINGDEATHS
SELECT * FROM COVIDDEATHSBYAGESEX
SELECT * FROM COVIDDEATHSBYSTATE

SELECT MAX(STARTDATE), MIN(STARTDATE), MAX(ENDDATE), MIN(ENDDATE)
FROM COVIDCONTRIBUTINGDEATHS
"2021-07-01", "2020-01-01", "2021-07-03", "2020-01-31"


SELECT DISTINCT GROUPING FROM COVIDCONTRIBUTINGDEATHS
ORDER BY GROUPING
"By Month", "By Total", "By Year"

states in US @ locaion https://worldpopulationreview.com/states/state-abbreviations


SELECT * FROM US_STATES

CREATE TABLE US_States (
  	State text,
	Abbrev text,
  	Code text
);

------------------

SELECT DISTINCT STATE, CONDITIONGROUP, SUM(DEATHS) TOTAL_DEATHS
FROM COVIDCONTRIBUTINGDEATHS 
WHERE STATE = 'Virginia' AND GROUPING LIKE 'By Total'
GROUP BY STATE, CONDITIONGROUP


SELECT DISTINCT COVIDCONTRIBUTINGDEATHS.STATE, CODE, CONDITIONGROUP, SUM(DEATHS) TOTAL_DEATHS
FROM COVIDCONTRIBUTINGDEATHS 
INNER JOIN US_STATES
ON COVIDCONTRIBUTINGDEATHS.STATE=US_STATES.STATE
WHERE COVIDCONTRIBUTINGDEATHS.STATE = 'Virginia' AND GROUPING LIKE 'By Total'
GROUP BY COVIDCONTRIBUTINGDEATHS.STATE, CODE, CONDITIONGROUP


SELECT DISTINCT COVIDCONTRIBUTINGDEATHS.STATE, CODE, CONDITIONGROUP, SUM(DEATHS) TOTAL_DEATHS
FROM COVIDCONTRIBUTINGDEATHS 
INNER JOIN US_STATES
ON COVIDCONTRIBUTINGDEATHS.STATE=US_STATES.STATE
WHERE GROUPING LIKE 'By Total'
GROUP BY COVIDCONTRIBUTINGDEATHS.STATE, CODE, CONDITIONGROUP
(612 ROWS)

-----------------

SELECT MAX(ASOFDATE), MIN(ASOFDATE), MAX(CREATED_AT), MIN(CREATED_AT)
FROM COVIDDEATHSBYSTATE
"2021-07-05", "2020-01-22", "2021-07-06", "2020-01-23"



SELECT DISTINCT STATE, TOT_CASES, NEW_CASE, TOT_DEATH, NEW_DEATH
FROM COVIDDEATHSBYSTATE
ORDER BY STATE

SELECT MAX(ASOFDATE), MIN(ASOFDATE), MAX(CREATED_AT), MIN(CREATED_AT)
FROM COVIDDEATHSBYSTATE

SELECT DISTINCT COVIDDEATHSBYSTATE.STATE CODE, US_STATES.STATE, SUM(TOT_CASES) TOTAL_CASES, SUM(NEW_CASE) NEW_CASES, SUM(TOT_DEATH) TOTAL_DEATHS, SUM(NEW_DEATH) NEW_DEATHS
FROM COVIDDEATHSBYSTATE
INNER JOIN US_STATES
ON COVIDDEATHSBYSTATE.STATE=US_STATES.CODE
GROUP BY COVIDDEATHSBYSTATE.STATE, US_STATES.STATE
(51 ROWS)

--------------------

CREATE TABLE JAN2020_JUL2021_COVIDDEATHS_BYSTATE (
  	CODE text,
	STATE text,
  	TOTAL_CASES INTEGER,
	NEW_CASES INTEGER,
	TOTAL_DEATHS INTEGER,
	NEW_DEATHS INTEGER
);


CREATE TABLE JAN2020_JUL2021_COVIDDEATH_CONTRIBUTERS (
  	CODE text,
	STATE text,
  	CONDITIONGROUP text,
	TOTAL_DEATHS INTEGER
);


SELECT * FROM JAN2020_JUL2021_COVIDDEATH_CONTRIBUTERS

INSERT INTO JAN2020_JUL2021_COVIDDEATH_CONTRIBUTERS
SELECT DISTINCT COVIDCONTRIBUTINGDEATHS.STATE, CODE, CONDITIONGROUP, SUM(DEATHS) TOTAL_DEATHS
FROM COVIDCONTRIBUTINGDEATHS 
INNER JOIN US_STATES
ON COVIDCONTRIBUTINGDEATHS.STATE=US_STATES.STATE
WHERE GROUPING LIKE 'By Total'
GROUP BY COVIDCONTRIBUTINGDEATHS.STATE, CODE, CONDITIONGROUP


SELECT * FROM JAN2020_JUL2021_COVIDDEATHS_BYSTATE

INSERT INTO JAN2020_JUL2021_COVIDDEATHS_BYSTATE
SELECT DISTINCT COVIDDEATHSBYSTATE.STATE CODE, US_STATES.STATE, SUM(TOT_CASES) TOTAL_CASES, SUM(NEW_CASE) NEW_CASES, SUM(TOT_DEATH) TOTAL_DEATHS, SUM(NEW_DEATH) NEW_DEATHS
FROM COVIDDEATHSBYSTATE
INNER JOIN US_STATES
ON COVIDDEATHSBYSTATE.STATE=US_STATES.CODE
GROUP BY COVIDDEATHSBYSTATE.STATE, US_STATES.STATE

--------------------------
cdc_database_cleaned.csv

case_month                      10
res_state                       41
age_group                        4
sex                              3
race                             6
ethnicity                        2
exposure_yn                      1
current_status                   2
symptom_status                   3
hosp_yn                          2
icu_yn                           2
death_yn                         2
underlying_conditions_yn         2

COVID_CDC_DATA
--------------------

## Dashboard
TABLEAU