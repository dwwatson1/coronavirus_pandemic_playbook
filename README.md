# Coronavirus Pandemic Playbook

## Machine Learning

The preliminary data used is a subset of our main data table wtih data from the state of Maryland. We chose one state to work with for the preliminary data as it is significantly smaller, which will help the model run faster for the testing phase. With the model working with the Maryland data, we can reasonably assume that the model should work with the whole dataset. For the preliminary data, we used features from age, gender, and race, hospitalization, ICU, and underlying conditions. This was most efficient for the preliminary model because these features were already included in the main dataset. 

The model we chose to use is a supervised random forest model. We chose supervised machine learning because we have labeled data (our features in tabular form) and outputs (whether someone has COVID or not). The input data, or our features, has a paired outcome which is plugged in to train the model to predict outcomes. Supervised machine learning models have target variables, or variables about which we want to gain a deeper understanding, which in our case is whether or not a person has COVID. We chose a random forest algorithm because they can handle a lot of input variables of which we will have many, it can account for null values which there are quite a bit in our base dataset, it can run efficiently on large datasets (the original dataset before transformation had 27 million rows), and most importantly random forest models can be used to rank the importance of input variables. This fits the question we are trying to answer perfectly - what are the top factors that influence the spread of COVID? A random forest model will help us rank the most influential factors. While a large number of trees in a random forest algorithm can be slow requiring a lot of computational power and resources, the advantages outweigh the disadvantages.

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
X = 

