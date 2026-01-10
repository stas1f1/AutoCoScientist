# ITO-5212-RecSys

## Overview

### Description

This dataset for this task is collected from Google reads in which people rate and review the books that they read. The task is to predict the ratings of the books for a given user.

### Data

The user-item interaction data is the main data for this challenge. This data is further split into training and test sets.

- train.csv. The training dataset contains a set of user_item ratings between users and items. The users explicitly rated the items that they interacted with between 1 to 5.
- test.csv. Each user is provided with a list of items in the test dataset, for each user, you will need to predict the ratings for all the items in their list.

Please train your recommender systems and generate the outputs for the test data.

### Rules

- One account per participant. You cannot sign up on Kaggle using multiple accounts.
- No private code or data sharing among students
- You may submit a maximum of 10 entries per day.

### Evaluation

The metric to be used in this challenge is the Mean Absolute Error (MAE). A brief description of MAE can be found [here](https://en.wikipedia.org/wiki/Mean_absolute_error).

### Submission Format

For every user-item pair in the test set, you need to provide a predicted rating. The submission file should have the following format (see `sample_submission.csv` file under the Data):

```csv
ID, rating
100000,2
100001,1
...
```

### Citation

Ehsan Shareghi and Jordan M. ITO-5212-RecSys. https://kaggle.com/competitions/ito-5212-rec-sys, 2024. Kaggle.

## Dataset Description

These files are essential files:

- train.csv - the training set
- test.csv - the test set
- sample_solution.csv - a sample submission file in the correct format

Data fields in the training, validation, and test data:

- user_id - an anonymous id unique to a given customer
- item_id - the id of an item
- rating -the interaction between a user and an item, ranges between 1-5
- book_name: the name of the book
- ID: this column only exists in test data and is there to handle Kaggle constraints. Avoid making any changes to this column as it is the main identifier when we mark your submission.
