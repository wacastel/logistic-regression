import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack

# 1. Load the data and clean the 'author' column
df = pd.read_csv('friends_quotes.csv')
df['author'] = df['author'].str.title()

# --- THE FIXES ---

# BUG FIX 1: Calculate the previous quote BEFORE filtering out minor characters
df['prev_quote'] = df['quote'].shift(1).fillna('')

# Filter for only the main 6 characters AFTER the shift
main_characters = ['Rachel', 'Ross', 'Monica', 'Chandler', 'Joey', 'Phoebe']
df_main = df[df['author'].isin(main_characters)].copy()

# BUG FIX 2: Add a 'prev_' prefix to every word in the previous quote
def prefix_words(text):
    return ' '.join(['prev_' + str(w) for w in str(text).split()])

df_main['prev_quote_prefixed'] = df_main['prev_quote'].apply(prefix_words)

# Combine the prefixed previous quote with the current quote
df_main['combined_text'] = df_main['prev_quote_prefixed'] + " " + df_main['quote']

# Add Structural Features
df_main['word_count'] = df_main['quote'].apply(lambda x: len(str(x).split()))
df_main['exclamation_count'] = df_main['quote'].apply(lambda x: str(x).count('!'))
df_main['question_count'] = df_main['quote'].apply(lambda x: str(x).count('?'))

# --- TROUBLESHOOTING PRINT STATEMENTS ---
print("\n--- FEATURE ENGINEERING SANITY CHECK ---")
sample = df_main[['author', 'prev_quote', 'quote', 'combined_text']].iloc[10]
print(f"AUTHOR: {sample['author']}")
print(f"RAW PREVIOUS QUOTE: {sample['prev_quote']}")
print(f"RAW CURRENT QUOTE: {sample['quote']}")
print(f"COMBINED (PREFIXED) TEXT: \n{sample['combined_text']}\n")
print("-" * 40)

# Isolate features and labels
X_text = df_main['combined_text']
X_struct = df_main[['word_count', 'exclamation_count', 'question_count']]
y = df_main['author']

# Perform the Train/Test Split
X_text_train, X_text_test, X_struct_train, X_struct_test, y_train, y_test = train_test_split(
    X_text, X_struct, y, test_size=0.20, random_state=42
)

# Vectorize the Text Features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
X_train_text_vec = vectorizer.fit_transform(X_text_train)
X_test_text_vec = vectorizer.transform(X_text_test)

# Scale the Structural Features
scaler = StandardScaler()
X_train_struct_scaled = scaler.fit_transform(X_struct_train)
X_test_struct_scaled = scaler.transform(X_struct_test)

# Combine the sparse text matrix with the SCALED dense structural features
X_train_final = hstack([X_train_text_vec, X_train_struct_scaled])
X_test_final = hstack([X_test_text_vec, X_test_struct_scaled])

# Initialize and train the Logistic Regression model
print("\nTraining the updated Logistic Regression model...")
model = LogisticRegression(max_iter=1000) 
model.fit(X_train_final, y_train)

# Make predictions and evaluate
predictions = model.predict(X_test_final)
accuracy = accuracy_score(y_test, predictions)

print(f"\nNew Model Accuracy: {accuracy * 100:.2f}%")
print("\n--- Classification Report ---")
print(classification_report(y_test, predictions))

# ==========================================
# --- INTERACTIVE PREDICTION LOOP ---
# ==========================================

def predict_custom_quote(prev_line, current_line, model, vectorizer, scaler):
    # 1. Prefix the previous line
    prefixed_prev = ' '.join(['prev_' + str(w) for w in str(prev_line).split()])
    combined_text = prefixed_prev + " " + current_line
    
    # 2. Extract Structural Features
    word_count = len(str(current_line).split())
    exc_count = str(current_line).count('!')
    q_count = str(current_line).count('?')
    
    struct_df = pd.DataFrame([[word_count, exc_count, q_count]], 
                             columns=['word_count', 'exclamation_count', 'question_count'])
    
    # 3. Vectorize and Scale
    text_vec = vectorizer.transform([combined_text])
    struct_scaled = scaler.transform(struct_df)
    
    # 4. Combine Features
    final_features = hstack([text_vec, struct_scaled])
    
    # 5. Predict
    prediction = model.predict(final_features)[0]
    probabilities = model.predict_proba(final_features)[0]
    classes = model.classes_
    
    # Sort probabilities from highest to lowest
    prob_dict = {classes[i]: probabilities[i] for i in range(len(classes))}
    sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    
    return prediction, sorted_probs

print("\n" + "="*60)
print("☕ WELCOME TO CENTRAL PERK: THE INTERACTIVE MODEL ☕")
print("="*60)
print("Type a previous line and a current line to see who the model thinks said it!")
print("Type 'quit' at any prompt to exit.\n")
print("--- Sample Inputs to Try ---")
print("1) Ross:     [Prev] We are practically broken up anyway. -> [Quote] We were on a break!")
print("2) Joey:     [Prev] Hey, Joey, how are you doing?        -> [Quote] How you doin'?")
print("3) Chandler: [Prev] Look at this giant dinosaur.         -> [Quote] Could that BE any bigger?")
print("4) Phoebe:   [Prev] What is that smell?                  -> [Quote] Smelly cat, smelly cat, what are they feeding you?")
print("5) Monica:   [Prev] I just cleaned the kitchen.          -> [Quote] I know!")
print("-" * 60 + "\n")

while True:
    prev_input = input("Enter the PREVIOUS line (or 'quit'): ")
    if prev_input.lower() == 'quit':
        break
        
    quote_input = input("Enter the CURRENT quote (or 'quit'): ")
    if quote_input.lower() == 'quit':
        break
        
    if not quote_input.strip():
        print("Please enter a valid quote!\n")
        continue

    # Run the prediction
    winner, probabilities = predict_custom_quote(prev_input, quote_input, model, vectorizer, scaler)
    
    print("\n" + "*"*40)
    print(f"  🏆 PREDICTION: {winner.upper()} 🏆")
    print("*"*40)
    print("Confidence Breakdown:")
    for char, prob in probabilities:
        print(f" - {char}: {prob*100:.2f}%")
    print("\n" + "-"*60 + "\n")

print("Thanks for visiting Central Perk!")
