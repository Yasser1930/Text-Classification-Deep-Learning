# Text Classification with Deep Learning

## Project Overview

This project focuses on developing a text classification system to assess user responses to a therapy chatbot. The goal is to identify whether user responses should be "flagged" or "not flagged" based on content indicators, which helps determine whether the user should continue the conversation with the bot or be referred for further assistance.

We employ a variety of machine learning and deep learning techniques to achieve accurate classification. This project includes:
- **Preprocessing** text data to remove noise and prepare it for modeling.
- **Feature extraction** through TF-IDF and embedding layers.
- **Model training** using traditional ML models and advanced deep learning architectures.
- **Evaluation metrics** to assess each model's performance and suitability.

The classification outcomes of this project can be applied to other contexts where determining user sentiment or intent is essential, providing a scalable solution for risk detection in mental health applications.

## Dataset

The dataset, `Sheet_1.csv`, contains **80 user responses** in the `response_text` column. Each user response was generated after the bot asked: *"Describe a time when you have acted as a resource for someone else."*

- **Classes**:
  - **Not Flagged**: Responses that do not raise any concerns, allowing the user to continue the conversation with the bot.
  - **Flagged**: Responses that indicate potential distress or risk, triggering a referral to a help resource.

- **Columns**:
  - **response_id**: Unique identifier for each response.
  - **class**: Target variable, where each entry is labeled as `flagged` or `not flagged`.
  - **response_text**: The user’s actual response to the bot’s prompt.

This dataset structure allows us to explore the relationship between language use and potential risk indicators, supporting proactive assistance based on response content.

## Preprocessing

The text preprocessing pipeline includes the following steps to ensure that the dataset is clean, relevant, and ready for modeling:

1. **Tokenization**: The `response_text` column is tokenized, breaking down each response into individual words.
   
2. **Stopword Removal**: Commonly used English words (e.g., "the", "is") are removed to reduce noise, focusing on words that provide more significant context.

3. **Punctuation Removal**: Punctuation is removed from the tokenized text to eliminate unnecessary symbols, which reduces dimensionality and improves model performance.

4. **TF-IDF Vectorization**: Text responses are converted into numerical format using Term Frequency-Inverse Document Frequency (TF-IDF), capturing the importance of each word relative to the entire dataset.

5. **Embedding and Padding for RNN Models**: For deep learning models using recurrent architectures, text data is tokenized, converted to integer sequences, and padded to a uniform length, enabling sequential models to process text consistently.

6. **Class Balancing with SMOTE**: The Synthetic Minority Oversampling Technique (SMOTE) is applied to balance the classes, ensuring that the model learns equally from both `flagged` and `not flagged` classes.

## Models

We explore both traditional machine learning models and advanced deep learning architectures to classify responses effectively:

1. **Traditional Machine Learning Models**:
   - **K-Nearest Neighbors (KNN)**: A straightforward approach that classifies responses based on the most similar examples in the training set.
   - **Random Forest**: An ensemble method that builds multiple decision trees, providing robustness against overfitting and variability.
   - **Naive Bayes**: A probabilistic model well-suited to text classification, leveraging conditional probabilities for word occurrence.
   - **Support Vector Machine (SVM)**: A linear classifier that maximizes margin separation, which helps in distinguishing between flagged and non-flagged classes.

2. **Deep Learning Architectures**:
   - **Multilayer Perceptron (MLP)**: A dense neural network model trained on TF-IDF features. It uses fully connected layers and dropout to classify the responses without leveraging word order.
   - **GRU Model**: A Gated Recurrent Unit model with embedding layers, which captures sequential information, making it well-suited to capturing patterns in text.
   - **LSTM Model**: A Long Short-Term Memory network that captures long-term dependencies within text data, particularly useful for nuanced language analysis in sequential data.

These diverse models allow for a thorough comparison across traditional and deep learning approaches to text classification, with each model evaluated for its ability to handle language patterns in the dataset.

## Evaluation

To assess the models, we use multiple metrics and visualizations to ensure an in-depth evaluation:

1. **Classification Report**:
   - Provides precision, recall, and F1-score for each class.
   - Helps to understand the model's performance in accurately identifying both `flagged` and `not flagged` responses.
   
2. **Accuracy Score**:
   - Measures the overall correctness of predictions.
   - Allows for quick comparisons across models to assess overall fit to the dataset.

3. **ROC-AUC Score**:
   - Area Under the Receiver Operating Characteristic Curve (ROC-AUC) shows the model’s ability to distinguish between the two classes.
   - Visualized through ROC curves, where models closer to the top-left corner perform better at separating classes.

4. **ROC Curves**:
   - ROC curves are plotted for each model, showing the trade-off between the true positive rate and false positive rate.
   - These curves provide a visual representation of model discrimination power and help compare model performances at various thresholds.

Each model's performance is thoroughly evaluated to determine the most effective approach for this text classification task, informing decisions for further improvement or deployment.
