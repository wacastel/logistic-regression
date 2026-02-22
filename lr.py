import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the data
df = pd.read_csv('friends_quotes.csv')

# 2. Clean the 'author' column (convert all names to Title Case)
df['author'] = df['author'].str.title()

# 3. Filter for only the main 6 characters
main_characters = ['Rachel', 'Ross', 'Monica', 'Chandler', 'Joey', 'Phoebe']
df_main = df[df['author'].isin(main_characters)].copy()

# 4. Isolate our features (X) and labels (y)
X = df_main['quote']
y = df_main['author']

# 5. Perform the Train/Test Split (80% training, 20% testing)
# Using random_state ensures we get the exact same random split every time we run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")

# 6. Initialize the TF-IDF Vectorizer
# stop_words='english' automatically removes common words like 'the', 'is', 'and'
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# 7. Fit the vectorizer on the training data, and transform both train and test sets
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print(f"\nShape of our vectorized training data: {X_train_vectorized.shape}")

# 8. Initialize and train the Logistic Regression model
print("\nTraining the Logistic Regression model... (This might take a few seconds)")
# We increase max_iter to give the math enough time to settle on the best weights
model = LogisticRegression(max_iter=1000) 
model.fit(X_train_vectorized, y_train)

# 9. Make predictions on the unseen testing data
predictions = model.predict(X_test_vectorized)

# 10. Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\n--- Classification Report ---")
print(classification_report(y_test, predictions))
