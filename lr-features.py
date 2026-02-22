import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler  # <-- NEW IMPORT
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack

# 1. Load the data and clean the 'author' column
df = pd.read_csv('friends_quotes.csv')
df['author'] = df['author'].str.title()

# Filter for only the main 6 characters
main_characters = ['Rachel', 'Ross', 'Monica', 'Chandler', 'Joey', 'Phoebe']
df_main = df[df['author'].isin(main_characters)].copy()

# --- FEATURE ENGINEERING ---

# 2. Add the Context: Shift the 'quote' column down by 1 to get the previous line
df_main['prev_quote'] = df_main['quote'].shift(1).fillna('')
df_main['combined_text'] = df_main['prev_quote'] + " " + df_main['quote']

# 3. Add Structural Features
df_main['word_count'] = df_main['quote'].apply(lambda x: len(str(x).split()))
df_main['exclamation_count'] = df_main['quote'].apply(lambda x: str(x).count('!'))
df_main['question_count'] = df_main['quote'].apply(lambda x: str(x).count('?'))

# 4. Isolate features and labels
X_text = df_main['combined_text']
X_struct = df_main[['word_count', 'exclamation_count', 'question_count']]
y = df_main['author']

# 5. Perform the Train/Test Split
X_text_train, X_text_test, X_struct_train, X_struct_test, y_train, y_test = train_test_split(
    X_text, X_struct, y, test_size=0.20, random_state=42
)

# 6a. Vectorize the Text Features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
X_train_text_vec = vectorizer.fit_transform(X_text_train)
X_test_text_vec = vectorizer.transform(X_text_test)

# --- THE FIX: FEATURE SCALING ---
# 6b. Scale the Structural Features
scaler = StandardScaler()
# We only FIT the scaler on the training data to prevent data leakage!
X_train_struct_scaled = scaler.fit_transform(X_struct_train)
X_test_struct_scaled = scaler.transform(X_struct_test)

# 7. Combine the sparse text matrix with the newly SCALED dense structural features
X_train_final = hstack([X_train_text_vec, X_train_struct_scaled])
X_test_final = hstack([X_test_text_vec, X_test_struct_scaled])

print(f"New shape of our training data: {X_train_final.shape}")

# 8. Initialize and train the Logistic Regression model
print("\nTraining the updated Logistic Regression model...")
# Since the data is scaled, we can lower max_iter back to 1000
model = LogisticRegression(max_iter=1000) 
model.fit(X_train_final, y_train)

# 9. Make predictions and evaluate
predictions = model.predict(X_test_final)
accuracy = accuracy_score(y_test, predictions)

print(f"\nNew Model Accuracy: {accuracy * 100:.2f}%")
print("\n--- Classification Report ---")
print(classification_report(y_test, predictions))
