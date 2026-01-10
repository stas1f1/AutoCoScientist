# Task

Predict passenger survival based on tabular data.

# Metric

Classification accuracy.

# Submission Format

The submission format for the competition is a csv file with the following format:

```
PassengerId,Survived
892,0
893,1
894,0
Etc.
```

# Dataset

- **train.csv** - Personal records for about two-thirds (~891) of the passengers, to be used as training data.
    - `PassengerId` - A unique Id for each passenger.
    - `Pclass` - Ticket class. 1 = 1st, 2 = 2nd, 3 = 3rd.
    - `Name` - The first and last names of the passenger.
    - `Sex` - Sex of the passenger.
    - `Age` - Age in years.
    - `SibSp` - # of siblings / spouses aboard.
    - `Parch` - # of parents / children aboard.
    - `Ticket` - Ticket number.
    - `Fare` - Passenger fare.
    - `Cabin` - Cabin number.
    - `Embarked` - Port of Embarkation. C = Cherbourg, Q = Queenstown, S = Southampton.
    - `Survived` - Whether the passenger survived. This is the target, the column you are trying to predict (0 = No, 1 = Yes).
- **test.csv** - Personal records for the remaining one-third (~418) of the passengers, to be used as test data. Your task is to predict the value of `Survived` for the passengers in this set.
- **sample_submission.csv** - A submission file in the correct format.
    - `PassengerId` - Id for each passenger in the test set.
    - `Survived` - The target. For each passenger, predict either 0 or 1.
