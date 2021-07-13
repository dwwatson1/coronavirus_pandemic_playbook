# Module 20

# Coronavirus Pandemic Playbook
## Group 4
	Leggett, Michael
	Essilfie-Bondzie, DINAH
	Teamir rezene, Yodit
	Ellerbe, Kimberly
	Gross, Jack
	Watson, David

					
----------
# Technologies Used

## Machine Learning
We will use Scikit-learn machine learning library for Python to reshape, stratify, split, test, train …. our data.

## Data Cleaning and Analysis
Initial table cleaning is done with Pandas in Colab
Where the fields with the required data is picked and all others dropped.


## Database Storage
The SQL database tool PGAdmin, uses postgreSQL to extract, organize and retrieve our data.

Had to have a table of US States because the data we a have uses different types of states.
SELECT * FROM US_STATES

CREATE TABLE US_States (
  	State text,
	Abbrev text,
  	Code text
);

SQL Schema:

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

Data intput into the SQL tables

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

## ERD
https://app.quickdatabasediagrams.com/#/

## Dashboard
TABLEAU
https://public.tableau.com/app/profile/dinah.bondzie/viz/Coronavirus_Pandamic_KnowHows/Sheet1

https://public.tableau.com/app/profile/dinah.bondzie/viz/COVID_Pandemic_ContributingFactors/Sheet1



end.
