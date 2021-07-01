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

### Reason for Selecting Topic

By creating a machine-learning model to predict the spread of respiratory viral infection like COVID-19, we can develop a pandemic playbook to help policy makers and business leaders prepare for a future pandemic/epidemic and respond appropriately.  

## Data Sources for Project
[John Hopkins Coronavirus Data](https://coronavirus.jhu.edu/data/new-cases-50-states)

[U.S. Census Data](https://www.census.gov/data/developers/data-sets.html)

[Additional data source that we are considering](https://docs.google.com/document/d/10i01u6oQAUVCbk5VTL6G0rIsTF9JlO1I90XTCDXWTCA/edit)

[Another possible data APIs]. (https://blogs.mulesoft.com/dev-guides/track-covid-19/)

## Questions to Investigate During Project

1. What is the population per state at the beginning and end of the pandemic?
2. How has race played a role in the spread of the COVID-19 pandemic?
3. Could the level of poverty and inequality affect the spread of COVID-19? If how does it affect? Adverseley?
4. Did having medical insurance play a role in the cure and deaths?
5. What were the top 5 MSAs (Metropolitan Statistical Areas) impacted by Covid-19? (Def: MSA is a geographical region with a relatively high population density at its core and close economic ties throughout the area.)
6. What were the top 20 uSAs (Micropolitan Statistical Areas) impacted by Covid-19?
7. During periods of Covid-19 case spikes, were there geographical or state areas that trended with these spikes?
8. Did political affiliation of areas have an influence on the number of Covid cases prior to vaccine distribution?
 
## Machine Learning
We have identified a linear regression model as the best model in order to complete our project. The inputs for the model will be covid cases by state, gender, age, weight, race, political party, religious affiliation, income level, and population density. We will run the model with the hopes of identifying the largest factors that played a role in the spread of covid-19. We will be considering the R squared value when running our model in order to consider if the model is well fitted. 
