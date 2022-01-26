#!/usr/bin/env python
# coding: utf-8

# # BAIT 509 Assignment 1: An introduction to Decision Trees, $k$-NN, Cross-validation and ML Fundamentals

# ## Introduction and learning goals <a name="in"></a>
# <hr>
# 
# In this assignment, you will work on the following:
# 
# - use the `fit` and `predict` paradigms in `sklearn`.
# - use the `score` method in `sklearn` to calculate classification accuracy. 
# - use `train_test_split` for data splitting and explain the importance of shuffling during data splitting. 
# - build a decision tree classifier on a real-world dataset.
# - build a $k$-nn classifier and explore different hyperparameters.

# ### Exercise 1: Decision trees with a toy dataset <a name="1"></a>
# <hr>
# 
# Suppose you have three different job offers with comparable salaries and job descriptions. You want to decide which one to accept, and you want to make this decision based on which job is likely to make you happy. Being a very systematic person, you come up with three features associated with the offers, which are important for your happiness: whether the colleagues are supportive, work-hour flexibility, and whether the company is a start-up or not (the columns `Supportive`, `Flexible` and `Startup` respectively). 

# In[1]:


import pandas as pd

offer_data = {
    # Features
    "Supportive": [1, 0, 0],
    "Flexible": [0, 0, 1],
    "Startup": [0, 1, 1],
    # Target
    "target": ["?", "?", "?"],
}

offer_df = pd.DataFrame(offer_data)
offer_df


# Next, you ask the following questions to some of your friends (who you think have similar notions of happiness) regarding their jobs:
# 
# 1. Do you have supportive colleagues? (1 for 'yes' and 0 for 'no')
# 2. Do you have flexible work hours? (1 for 'yes' and 0 for 'no')
# 3. Do you work for a start-up? (1 for 'start up' and 0 for 'non start up')
# 4. Are you happy with your job? (happy or unhappy)
# 
# You get the following data from this survey. You want to train a machine learning model using this data and then use this model to predict which job is likely to make you happy. 

# In[2]:


happiness_data = {
    # Features
    "Supportive": [1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    "Flexible": [1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    "Startup": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    # Target
    "target": [
        "happy",
        "happy",
        "happy",
        "unhappy",
        "unhappy",
        "happy",
        "happy",
        "unhappy",
        "unhappy",
        "unhappy",
    ],
}

train_df = pd.DataFrame(happiness_data)
train_df


# ### 1.1 Decision stump by hand 
# rubric={autograde:2}
# 
# If you manually built a decision stump (decision tree with only 1 split) by splitting on the condition `Supportive == 1` by hand, how would you predict each of the employees? 
# 
# Save your prediction for each employee as a string element in a list named `predict_employees`. 
# Example:
# 
# ```
# predict_employees = ['happy', 'unhappy', 'unhappy',  'unhappy', 'unhappy', 'happy', 'happy', 'happy',  'unhappy',  'unhappy'] 
# ```
# 
# (Note: you do not need to use a model here. By looking at the target column and the feature `Supportive` what rows would you predict to have which labels?) 

# In[3]:


# Replace the `...` with your list of hapiness predictions
predict_employees = ['happy', 'happy', 'happy', 'unhappy', 'unhappy', 'happy', 'happy', 'unhappy', 'happy', 'unhappy']


# ### 1.2 Decision stump accuracy
# 
# rubric={autograde:2}
# 
# What training accuracy would you get with this decision stump above?
# 
# Save the accuracy as a decimal in an object named `supportive_colleagues_acc`. 

# In[4]:


supportive_colleagues_acc = 0.9


# ### 1.3 Create `X`, `y`
# rubric={mechanics:2}
# 
# Recall that in `scikit-learn` before building a classifier we need to create `X` (features) and `y` (target). 
# 
# **Your tasks:**
# 
# From `train_df`, create `X` and `y`; save them in objects named `X` and `y`, respectively. 

# In[5]:


X = train_df.drop(columns=["target"])
y = train_df["target"]


# ### 1.4 `fit` a decision tree classifier 
# rubric={accuracy:2}
# 
# The idea of a machine learning algorithm is to *fit* the best model on the given training data, `X` (features) and `y` (their corresponding targets) and then using this model to *predict* targets for new examples. 
# 
# **Your tasks:**
# 
# Build a decision tree named `toy_tree` and fit it on the toy data using `sklearn`'s [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html). Don't forget to make the necessary import(s) first.

# In[6]:


from sklearn.tree import DecisionTreeClassifier

toy_tree = DecisionTreeClassifier()
toy_tree.fit(X, y)


# ### 1.5 `score` 
# rubric={accuracy:2}
# 
# Score the decision tree on the training data (`X` and `y`).
# Save the results in an object named `toy_score` and output the score at the end of the cell.

# In[7]:


toy_score = toy_tree.score(X,y)
toy_score


# ### 1.6 Explain training score
# rubric={reasoning:2}
# 
# Do you get perfect training accuracy? Why or why not? 
# 
# ##### Answer:
# Becasue there is inconsistency in the dataset, as for rows "2" and "8" we have the same values (1, 0 , 1). However, the target is "happy" for row 2 while it is "unhappy" for row 8.
# 

# ### 1.7 Getting features
# 
# rubric={accuracy:2}
# 
# The first `offer_df` dataframe has no target values and we want to use the model we just made to make predictions. 
# Drop the column `target` from the object and rename this dataframe `test_X`. 

# In[8]:


test_X = offer_df.drop(columns=["target"]) 


# ### 1.8 `predict`
# rubric={accuracy:2}
# 
# Now make predictions on the jobs offered in `test_X`. Save the predictions in an object named `predicted`. 
# 

# In[9]:


predicted = toy_tree.predict(test_X)
predicted


# ### 1.9 Happy job
# rubric={reasoning:2}
# 
# Looking at the predictions, in which job you are likely to be happy?
# 
# ##### Answer:
# 
# I seem to be more likely to be happy in the first job based on the prediction as the train set results show unhappy for the other 2 situations, but have resulted in happiness for the first one.

# ## Exercise 2: Decision trees on a real dataset <a name="2"></a>
# <hr>

# ### Introducing the Spotify Song Attributes dataset
#  
# For the rest of the assignment, you'll be using Kaggle's [Spotify Song Attributes](https://www.kaggle.com/geomack/spotifyclassification/home) dataset.
# The dataset contains a number of features of songs from 2017 and **a binary target variable representing whether the user liked the song or not** (1 = liked, 0 = did not like). See the documentation of all the features [here](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/). The supervised machine learning task for this dataset is predicting  whether the user likes a song or not given a number of song features.  

# The starter code below reads the data CSV file into the notebook. make sure you named the csv file `spotify.csv`

# In[10]:


spotify_df = pd.read_csv("spotify.csv", index_col=0)
spotify_df


# ### 2.1 Split your data
# rubric={accuracy:2}
# 
# Split your `spotify_df` into your train and test splits.  Name the training data `train_df` and the testing data `test_df` using an 80/20 train to test split. Set your `random_state` to 77 to keep it consistent and facilitate grading.

# In[11]:


# Assign the splits to train_df and test_df

from sklearn.model_selection import train_test_split

# Split the dataset into 80% train and 20% test 
train_df, test_df = train_test_split(spotify_df, test_size = 0.2, train_size = 0.8, random_state = 77)


# ### 2.2 Explaining histograms 
# 
# rubric={reasoning:3}

# A good thing to do before starting to train our models
# would be to explore the features visually so that we have an idea of what the data looks like.
# It is often beneficial to view the distributions of data for each feature.
# 
# Create histograms for each of the features,
# showing the distribution for each target class.
# Based on these histograms which features and split values you think might be useful in differentiating the target classes?
# 
# ##### Answer:
# 
# Danceability, energy, and valence as they have different rages for the peak of the 2 target values to happen. Danceability have 2 distict peaks for 0 and 1 around 0.6 and 0.75 respectively. target 0 on Energy has some values on the lower end which is almost missing for target 1 on the same graph. The histogram for valence with target 0 is right-skewed while for target =1 it can be interpreted skewed a bit to the left.

# In[12]:


import altair as alt

alt.Chart(train_df.sort_values(by='target')).mark_bar(opacity=0.6).encode(
    alt.X(alt.repeat(), type='quantitative', bin=alt.Bin(maxbins=50)),
    alt.Y('count()', stack=None),
    alt.Color('target:N')
).properties(
    height=200
).repeat(
    ["acousticness", "danceability", "tempo", "instrumentalness", "energy", "valence"],
    columns=2
)


# ## Exercise 3: Cross-validation and model building <a name="3"></a>
# <hr>
# Recall that in machine learning what we care about is generalization; we want to build models that generalize well on unseen examples. One way to ensure this is by splitting the data into training data and test data, building and tuning the model only using the training data, and then doing the final assessment on the test data. 

# We've provided you with some starter code that separates `train_df` and `test_df` into their respective features and target objects. We removed the columns `song_title` and `artist` from the feature objects since they would need additional processing to be used in our model. 

# In[13]:


X_train = train_df.drop(columns = ['song_title', 'artist','target'])
y_train = train_df['target']
X_test = test_df.drop(columns = ['song_title', 'artist','target'])
y_test = test_df['target']


# ### 3.1 Building a Dummy Classifier
# rubric={accuracy:3}
# 
# Build a `DummyClassifier` using the strategy `most_frequent`.
# 
# Train it on `X_train` and `y_train`. Score it on the train **and** test sets.

# In[14]:


from sklearn.dummy import DummyClassifier

dummy_model = DummyClassifier(strategy="most_frequent")

dummy_model.fit(X_train, y_train)
dummy_model.score(X_train, y_train)


# ### 3.2 Building a Decision Tree Classifier
# rubric={accuracy:3}
# 
# Build a Decision Tree classifier without setting any hyperparameters. Cross-validate with the appropriate objects, passing `return_train_score=True` and setting the number of folds to 10. (See the note in lecture 2 for help).
# 
# Display the scores from `.cross_validate()` in a dataframe. 

# In[15]:


from sklearn.model_selection import cross_validate

dt_model = DecisionTreeClassifier()

dt_cv_scores = cross_validate(dt_model, X_train, y_train, cv=10, return_train_score=True)

pd.DataFrame(dt_cv_scores)


# ### Question 3.3 Decision Tree training and validation scores
# rubric={accuracy:1, reasoning:1}
# 
# What are the mean validation and train scores? In 1-2 sentences, explain your results. Is your model overfitting or underfitting? 
# 
# ##### Answer:
# 
# There is a significant difference between the mean score values for test and train sets; therefore, the model is probably overfitting. 

# In[16]:


dt_cv_means = pd.DataFrame(dt_cv_scores).mean()[-2:]
dt_cv_means


# ### 3.4 Building a $k$-NN Classifier
# rubric={accuracy:3}
# 
# Build a $k$-NN classifier using the default hyperparameters. Cross-validate with the appropriate objects, passing `return_train_score=True` and setting the number of folds to 10.
# 
# Display the scores from `.cross_validate()` in a dataframe. 

# In[17]:


from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier() #(n_neighbors=1)
knn_model.fit(X_train, y_train)

knn_cv_scores = cross_validate(knn_model, X_train, y_train, cv=10, return_train_score=True)

pd.DataFrame(knn_cv_scores)


# ### Question 3.5 $k$-NN training and validation scores 
# rubric={accuracy:1, reasoning:1}
# 
# What are the mean validation and train scores for your $k$-NN classifier? In 1-2 sentences, explain your results.
# 
# ##### Answer:
# 
# 

# In[18]:


knn_cv_mean = pd.DataFrame(knn_cv_scores).mean()[-2:]
knn_cv_mean


# ### 3.6 Compare the models
# rubric={reasoning:2}
# 
# In 1-2 sentences, compare the 3 models.
# 
# ##### Answer:
# 
# Decision tree classification generates the best results (nearly 70% accuracy on the test set). Knn is the second best with an accuracy around 57%. Dummy classifier has the worst performance as expected since it decides based on the most frequent target values.

# ## Exercise 4: Hyperparameters <a name="5"></a>
# <hr>
# 
# We explored the `max_depth` hyperparameter of the `DecisionTreeClassifier` in lecture 2 but in this assignment, you'll explore another hyperparameter, `min_samples_split`. See the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) for more details on this hyperparameter.

# ### 4.1 `min_samples_splits`
# rubric={accuracy:5}
# 
# Using 10-fold cross-validation and the training set only, find an appropriate value within the range 5 to 105 for the `min_samples_split` hyperparameter for a decision tree classifier.
# 
# For each `min_samples_split` value:
# 
# - Create a `DecisionTreeClassifier` object with the `min_samples_split` value.
# - Run 10-fold cross-validation with this `min_samples_split` using `cross_validate` to get the mean train and validation accuracies. Remember to use `return_train_score` argument to get the training score in each fold. 
# 
# In a pandas dataframe, for each `min_samples_split` show the mean train and cross-validation score. 
# 
# *Hint: We did something similar in lecture 2 (under **The "Fundamental Tradeoff" of Supervised Learning**) which you can refer to if you need help.* 

# In[19]:


results_dict = {"min_samples_split": list(), "mean_train_score": list(), "mean_cv_score": list()}

# Iterate over the hyperparameter values
for minss in range(2, 30):
    model = DecisionTreeClassifier(min_samples_split=minss)
    scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)
    results_dict["min_samples_split"].append(minss)
    results_dict["mean_cv_score"].append(scores["test_score"].mean())
    results_dict["mean_train_score"].append(scores["train_score"].mean())

results_df = pd.DataFrame(results_dict)
results_df


# ### 4.2 Plotting and interpreting
# rubric={accuracy:3, viz:1}
# 
# Using whatever tool you like for plotting,  make a plot with the `min_samples_split` of the decision tree on the *x*-axis and the accuracy on the train and validation sets on the *y*-axis. 
# 
# (Again we did this in lecture 2 if you need any guidance)

# In[20]:


import altair as alt

source = results_df.melt(id_vars=['min_samples_split'] , 
                              value_vars=['mean_train_score', 'mean_cv_score'], 
                              var_name='score_type', value_name='accuracy')
accuracy_plot = alt.Chart(source).mark_line().encode(
    alt.X('min_samples_split:Q', axis=alt.Axis(title="Min Sample Split Value")),
    alt.Y('accuracy:Q'),
    alt.Color('score_type:N', scale=alt.Scale(domain=['mean_train_score', 'mean_cv_score'],
                                           range=['teal', 'gold'])))

# Display the plot
accuracy_plot


# ### 4.3 Picking `min_samples_split`
# rubric={accuracy:1, reasoning:2}
# 
# Based on your results from 4.2, what `min_samples_split` value would you pick in your final model? In 1-2 sentences briefly explain why you chose this particular value.
# 
# ##### Answer:
# We should extract from the graph where the cross-validation score peaks? at what corresponding value of "min_samples_split"? That would be the optimum value for "min_samples_split"

# In[21]:


# Which value of min_samples_split is the best
max_val = results_df['mean_cv_score'].idxmax()
best_split = max_val + 2
# min_samples_split is looped starting "2"
# therefore the equivalent min_samples_split would be the max_val (which is the rwo number) plus 2
best_split


# ### 4.4 Final model
# rubric={accuracy:2,reasoning:1}
# 
# Train a decision tree classifier with the best `min_samples_split` using `X_train` and `y_train` and now carry out a final assessment by obtaining the test score on the test set.
# 
# ##### Answer:
# 
# The `min_samples_split` uses decision tree and has a prediction power closer to the normal decision tree. However, we can choose which values give the best predictions so the accuracy is notable.

# In[22]:


model = DecisionTreeClassifier(min_samples_split=13)

model.fit(X_train, y_train);
print("Score on test set: " + str(round(model.score(X_test, y_test), 2)))


# ### The end
