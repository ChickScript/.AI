Create NLP Pipeline for Model Training



# Create a pipeline for the deep learning model with custom preprocessing
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(preprocessor=preprocess_text)),  # Use custom preprocessing
    ('mlp', MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000))  # Multi-layer Perceptron
])

# Train the model
pipeline.fit(X_train, y_train)

# Function to evaluate the model
def evaluate_model():
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

# Call evaluation after training
evaluate_model()
