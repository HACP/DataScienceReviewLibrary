# Exploratory Data Analysis

The purpose of an Exploratory Data Analysis (EDA) is to learn as much as possible from the data in hand and identify relevant information, 
assess data quality and pinpoint issues, decide feature enginering approach, gather relevant information about potential models and build the
foundation of the data story. 

EDA helps understanding the real problem at hand. Often times, the need for a predictive model is only the surface od a much deeper need. The 
model can help improve the understanding of the problem. 

A few questions to consider when performing an EDA
- What is the format of the available data? Is is a rectangular table, in a csv or excel format? Is it unstructed data? How many files and 
  is the relationship between them? Can we build an Entity Relationship Diagram? 
- What is the the quality of the data? A few metrics to consider: number of features, number of rows, number of null values, ratio of null values
  relative to number of records, number of unique values, ratio of unique values relative to the total number of records
- What is the correlation between the features? If the problem requires a prediction of a target feature, what's the correlation between independent
  features and the target feature? 
- What is the information content of the features? Example: Shannon entropy per feature, Mutual Information.
