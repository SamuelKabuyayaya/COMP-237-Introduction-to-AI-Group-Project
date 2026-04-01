import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#Task 1: Loading the data and looking at it
df = pd.read_csv('Youtube02-KatyPerry.csv')

print("")
print("Looking at data")
print(df.head())

#Task 2 Basic Data Exploration
#2.1 Feature selection
df = df[['CONTENT', 'CLASS']]

#2.2 Checking if we deleted stuff we didnt need
print("")
print("Dataset after deleting columns")
print(df.head())

#2.3 Checking class distribution
print("")
print("Checking balance of classes")
counts = df['CLASS'].value_counts()
print("Normal messages: " + str(counts[0]))
print("Spam messages: " + str(counts[1]))

#2.4 Checking dataset shape
print("")
print("Dataset shape:")
print(str(df.shape[0]) + " rows")
print(str(df.shape[1]) + " columns")

#2.5 Checking for missing values
print("")
print("Checking for missing values")    
print(df.isnull().sum())

#Task 3: Data preparation for model building
#3.1 Creating the CounterVectorizer for The "Bag of Words" model
count_vectorizer = CountVectorizer()

#3.2 Transfoming text into numerical matrix 
X_counts = count_vectorizer.fit_transform(df['CONTENT'])
print("")
print("Bag of words preparation done.")

#Task 4: Initial features showcase
print("")
print("Initial Features:")

#4.1 The new data shape
print("")
print("New data shape of the matrix:" + str(X_counts.shape))

#4.2 Getting feature names
print("")
feature_names = count_vectorizer.get_feature_names_out()
print("Number of unique features:" + str(len(feature_names)))

#4.3 Showing some examples
print("")
print("Some example of 10 features from the vocabulary:")
print(feature_names[:10])

# 4.4 Showing a snippet of the numerical matrix
print("")
print("Numerical representation of the first comment:")
print(X_counts[0, :10].toarray())
