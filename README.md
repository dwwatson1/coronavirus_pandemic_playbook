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

### Audience

The National Governors Association has tasked our group, the **COVID-19 Rapid Response Group: Preparing for the Next Pandemic** will be presenting our pandemic playbook to at a COVID-19 response conference with the National Governors Association and to the United States Conference of Mayors. The two nonpartisan organizations comprised of governors from the U.S.'s 55 states and territories and mayors of U.S. cities with population of 30,000 or more. Typically these groups meet separately to share strategies across their jurisdictions, however, this conference is a rare opportunity for all executives to gather and learn how to better respond to the next pandemic and minimize the spread, deaths, and economic impact. 

### Why Should We Care?

As a capitalist society, some economic measures of success for the U.S. are jobs created and GDP growth. The COVID-19 brought the [deepest recession since the end of WWII](https://www.brookings.edu/research/social-and-economic-impact-of-covid-19/) as the global economy shrunk by 3.5% and [114 million](https://www.weforum.org/agenda/2021/02/covid-employment-global-job-loss/) people lost their jobs in 2020. The impact of this shock is likely to be felt for years to come.

[The Brookings Institute](https://www.brookings.edu/research/social-and-economic-impact-of-covid-19/) identified **state capacity** as one of three pre-existing conditions that amplified the impact of the shock. The COVID-19 crisis posed a critical challenge for policymakers as they needed to quickly reach workers and households during the abrupt economic crisis. There is evidence that if states were more prepared to handle a pandemic, economic performance would not have suffered as it did in 2020. Our nation's governors and mayors have the opportunity to learn where our countries weak points are that led to these incredible economic losses and mitigate them in a future pandemic. 

### Data Sources for Project
[John Hopkins Coronavirus Data](https://coronavirus.jhu.edu/data/new-cases-50-states)

[U.S. Census Data](https://www.census.gov/data/developers/data-sets.html)

[Additional data source that we are considering](https://docs.google.com/document/d/10i01u6oQAUVCbk5VTL6G0rIsTF9JlO1I90XTCDXWTCA/edit)

[Another possible data APIs](https://blogs.mulesoft.com/dev-guides/track-covid-19/)

[A database that we can consider] (https://covidtracking.com/data/download)


### Questions to Investigate During Project

1. What is the population per state at the beginning and end of the pandemic?
2. How has race played a role in the spread of the COVID-19 pandemic?
3. Could the level of poverty and inequality affect the spread of COVID-19? If so what is the impact?
4. Did having medical insurance play a role in the cure and deaths?
5. What were the top 5 MSAs (Metropolitan Statistical Areas) impacted by Covid-19? (Def: MSA is a geographical region with a relatively high population density at its core and close economic ties throughout the area.)
6. What were the top 20 uSAs (Micropolitan Statistical Areas) impacted by Covid-19?
7. During periods of Covid-19 case spikes, were there geographical or state areas that trended with these spikes?
8. Did political affiliation of areas have an influence on the number of Covid cases prior to vaccine distribution?
 
### Machine Learning
We have identified a multiple linear regression model as the best model in order to complete our project. The inputs for the model will be covid cases by state, gender, age, weight, race, political party, religious affiliation, income level, and population density. We are using a multiple linear regression model because we have multiple explanatory variables (the independent variables which are the earlier identified factors) and we want to know how strong the relationship is between these independent variables with our dependent variable, which is covid cases. We are using a linear model as opposed to a logisitical model because our dependent variable is continuous. We will run the model with the hopes of identifying the largest factors that played a role in the spread of covid-19. We will be considering the R squared value and the standardized coefficient when running our model in order to consider if the model is well fitted and compare the factors to see which play a larger role. The sample equation that we are using to build the model is below:

yi=β0+β1xi,1+β2xi,2+…+βkxi,k+ϵi.

From an early perspectivce we have identified:

yi = dependent variable—the number of COVID-19 cases

xi1 = explanatory variable-the number of men and women

xi2 = religion

xi3 = race

xi4 = political party affiliation

xi5 = income level

xi6 = health care professionals/doctors registered by state

### Communication Protocol 

We are utilizing the available and most suitable resources as our communication tools. Zoom and Slack. We are aiming to meet twice a week in addition to meeting and discussing over the regular virtual class hours. 
We have created a group and direct messages for group members in Slack, and we may use this channel for any cases of emergency. 

### Technologies Used

#### Database storage

We will use SQL database tool to extract, organize and retrieve our data.

#### Data Cleaning and Analysis
Pandas will be used to clean and transform the data and perform an exploratory analysis. 

#### Dashboard

We will create an HTML/CSS portfolio to showcase our project and Bootstrap components to polish and customize the portfolio. We will also use JavaScript functions to display dynamic and interactive dashboard. 

#### Machine Learning 

We will use Scikit-learn machine learning library for Python to reshape, stratify, split, test, train …. our data. 
