import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score

#Task 1: Loading the data and looking at it
df = pd.read_csv('Youtube02-KatyPerry.csv')

print("")
print("Looking at data")
print(df.head())

#Task 6: Shuffling the dataset
df = df.sample(frac=1)

print("")
print("Dataset shuffled.")
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

#Task 5: Downscaling the transformed data using tf-idf
#5.1 Creating the tf-idf transformer
tfidf_transformer = TfidfTransformer()

#5.2 Transforming the count matrix into tf-idf features
X_tfidf = tfidf_transformer.fit_transform(X_counts)

print("")
print("TF-IDF downscaling done.")

#5.3 Showing highlights of final features
print("")
print("Final Features:")

# Shape
print("New data shape after tf-idf: " + str(X_tfidf.shape))

# Example values
print("")
print("TF-IDF representation of first comment:")
print(X_tfidf[0, :10].toarray())

# Non-zero values
print("")
print("Number of non-zero values:")
print(X_tfidf.nnz)

# 7.1 Calculate the split index
# We take 75% of the total number of rows
train_size = int(0.75 * len(df))

# 7.2 Separating Features (X) and Target (y) X is our TF-IDF matrix, y is the CLASS column from our shuffled dataframe
X = X_tfidf
y = df['CLASS']

# 7.3 Manual splitting using indexing
# Training data 75%
X_train = X[:train_size]
y_train = y[:train_size]

# Testing data 25%
X_test = X[train_size:]
y_test = y[train_size:]

print("Manual Split Done")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size:  {X_test.shape[0]} samples")

# Showing separation
print("")
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape:   {y.shape}")

# 8.1 Initialize the Naive Bayes classifier. MultinomialNB often used for text classification
model = MultinomialNB()

# 8.2 Train (Fit) the model. We give it the training features (X_train) and the correct answers (y_train)
model.fit(X_train, y_train)

print("Model Training Fitting")
print("The Naive Bayes classifier has been trained on 75% of the data.")

# 9.1 Performing 5-fold cross validation. We use only the training data (X_train, y_train) as per instructions
scores = cross_val_score(model, X_train, y_train, cv=5)

print("5-Fold Cross-Validation Results")

# 9.2 Print individual scores for each fold
print(f"Accuracy for each fold: {scores}")

# 9.3 Printing the mean result (Average accuracy)
print(f"Mean Accuracy: {scores.mean():.4f}")
print(f"Standard Deviation: {scores.std():.4f}")