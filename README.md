# Practical Application III: Comparing Classifiers - Module 17.1

## About Dataset
This dataset is about the direct marketing campaigns, which aim to promote term deposits among existing customers, by a Portuguese banking institution from May 2008 to November 2010. It is publicly available in the UCI Machine Learning Repository, which can be retrieved from http://archive.ics.uci.edu/ml/datasets/Bank+Marketing#.

## Task Objectives
Direct marketing campaigns are an inexpensive means to increase sales; however, the performance of direct phone marketing is largely unknown.

In addition to performance, more information is need for improvement in efficiency of direct phone marketing. The goal is to examine the peformance of multiple classifier methods and find the one that returns the best metrics.

In this situation, there is a low cost or penalty for type-I errors i.e., false alarms. In this case we're targeting customers with phone calls who are likely to already purchase products. This is inefficient and maximizing precision will reduce costs and being a nuisance to buying customers. Type-II errors or missed detections is more detremental due to missing sales oppurtunities that would help. Here we wish to maximize recall score. Since there is a large class imbalance, accuracy score is not the best metric given that most of the feature space is of the class 0 or 'no sale'. We therefore optimize for maximum f1-score (combination of recall and precision) but utimately also want to pick methods that optimize recall.

Besides comparing the performance of the classifiers, we also wish to analize the imporatnce of the features and provide some interpretation so to improve the efficiency for future direct markinging methods.

## Model Comparisons
image.png

## Findings and Scope for Imprevements
PCA reduced the number of features needed to train classifiers mainly reducing the number of non-important features. In this assignment PCA uses different compoenents including all 19 principal compoenents for comparison of performance of the features in the model evaluation. There is a scope to imporove or try with more or different princiapal components for predictions and evaluate specific features for model improvement.

All models used standard process/steps and same dataset/split train and test datasets for better model comparison. All models used baseline GridSearchCV with mimimum hyperparameters for fitting training data and performed predictions, scores and comparisons.

KNeighborsClassifier and Decision Tree Classifier with GridSearchCV with minimal hyperparameters as baseline performed best with respect to accuracy, recall and f1 scores when compared to other classifiers in this assignment.

Performed fine tuning on top KNeighborsClassifier further using additional hyperparameters and the outcome has good scores with respect to 90.7% accuracy and 46.5% f1 scores. This fine tuned model run peformed much better than all other models and stood on top in the table. Please refer to the evidence in the table above.

There is a scope to perform model tuning with different combinations KNeighborsClassifier hyperparameters like algorithms, p and metric values for better predictions and scores.
