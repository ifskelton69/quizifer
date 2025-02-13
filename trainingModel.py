import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Read the file
with open("question_extraction_output.txt", "r") as file:
    lines = file.readlines()

# Extract word and score from each line
word_scores = {}
for line in lines:
    try:
        word, score = line.strip().split(": ", 1)  # Split only at the first colon
        word_scores[word] = float(score)
    except ValueError:
        print(f"Skipping invalid line: {line.strip()}")  # Log or handle lines that don't match the format

# Create a DataFrame from the word scores
df = pd.DataFrame(list(word_scores.items()), columns=['Word', 'Score'])

# Example label assignment logic (you can adjust this based on your requirements)
threshold = 0.5
df['Label'] = df['Score'].apply(lambda x: 1 if x > threshold else 0)

# Split the data into features (X) and labels (y)
X = df[['Score']]
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
