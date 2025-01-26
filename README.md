# Phase_4_Group_9_Project
## Overview

This project focuses on leveraging sentiment analysis, a natural language processing (NLP) technique, to classify tweets about Apple and Google products into positive, negative, or neutral categories. The goal is to provide actionable insights for businesses to improve customer experiences, refine marketing strategies, and enhance decision-making.

## Problem Statement

In the digital era, customer feedback is crucial for shaping business strategies. However, the vast amount of unstructured textual data from social media makes manual analysis inefficient. This project aims to develop a sentiment analysis model to automatically classify customer feedback, enabling businesses to:

- Pinpoint areas needing improvement.
- Customize marketing efforts based on customer sentiment.
- Track and manage brand reputation effectively.
  
## Project Objectives

### Primary Objective
- Develop an unsupervised machine learning model to classify tweets as positive, negative, or neutral.

### Secondary Objectives
- Identify emotions directed at specific brands or products.
- Preprocess and clean tweet text (e.g., remove hashtags, mentions, URLs).
- Identify the most positive and negative words associated with Apple and Google products.
- Provide actionable insights to improve customer satisfaction and marketing strategies.

## Methodology

The project follows a systematic approach:

1. **Text Preprocessing**:  
   - Normalization  
   - Tokenization  
   - Stop word removal  
   - Lemmatization  
   - Noise removal (e.g., URLs, hashtags)

2. **Model Selection**:  
   - Baseline models (e.g., Logistic Regression, Naive Bayes)  
   - Advanced models (e.g., Random Forest, Gradient Boosting)  
   - Deep learning models  

3. **Model Evaluation**:  
   Metrics include:  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  
   - Confusion matrix  

4. **Performance Benchmarking**:  
   Compare models to identify the best approach.

## Metrics of Success

To evaluate the sentiment analysis model's effectiveness, the following metrics are considered:

1. **Accuracy**:  
   - Measures the percentage of correctly classified sentiments.  
   - **Target**: 85% or higher.

2. **Precision**:  
   - Reflects the percentage of correctly predicted sentiments.  
   - **Target**: 80% or higher for each sentiment class (positive, negative, neutral).

3. **Recall**:  
   - Assesses the model's ability to identify all relevant sentiment instances.  
   - **Target**: 75% or higher for each sentiment class.

4. **F1-Score**:  
   - Combines precision and recall for a balanced performance measure.  
   - **Target**: 80% or higher overall.

5. **Confusion Matrix**:  
   - Visualizes true positives, true negatives, false positives, and false negatives.

6. **Business Impact**:  
   - Enhances customer satisfaction by addressing negative sentiments.  
   - Informs marketing strategies through trend analysis in positive feedback.

These metrics ensure reliability and actionable insights for improved customer experiences and business strategies.

## Data Understanding

The dataset, sourced from CrowdFlower, contains 9,093 tweets with three main columns:

- **tweet_text**: The original tweet.
- **Emotions_in_tweet_is_directed_at**: The product (e.g., Apple, Google) being discussed.
- **Is_there_an_emotion_directed_at_a_brand_or_product**: The emotion (positive, negative, neutral) associated with the tweet.

# Data Preparation

- **Handling Missing Values**: 
  - Rows with missing `tweet_text` were dropped.
  - Missing values in the `Emotions_in_tweet_is_directed_at` column were imputed.
    ![image](https://github.com/user-attachments/assets/227b431a-e259-4303-8ced-74c70d28d804)

- **Data Cleaning**: 
  - Removed duplicates. 
  - Standardized text by converting to lowercase and removing special characters. 
  - Extracted mentions and hashtags.
    ![image](https://github.com/user-attachments/assets/7d51e74a-891e-4372-a6b2-6d8b04c22d41)

- **Feature Engineering**: 
  - Created new columns for cleaned text, mentions, and hashtags.

# Exploratory Data Analysis (EDA)

## Word Clouds
Visualized the most frequent **positive** and **negative** words for Apple and Google:

  
  - **Most positive words for Apple**.
  
![image](https://github.com/user-attachments/assets/95eeb4ba-ab6e-4e7e-b1bd-50b4ee9d0283)

- **Most negative words for Apple**
  
![image](https://github.com/user-attachments/assets/eecda392-b847-4f26-ad7e-e6435e41327a)


- **Most positive words Google**

  ![image](https://github.com/user-attachments/assets/4b67d570-7017-4909-ad63-1281cb9afc75)

  -**Most negative words Google**

  ![image](https://github.com/user-attachments/assets/838d194d-96e5-4964-af4e-ddd022f65c0a)

### Insights
- Highlighted key terms associated with positive and negative sentiments.
- Provided a visual representation of word prominence and patterns.

---

## Frequency Distribution
Identified **common words** associated with:
- **Positive Sentiments**
- **Negative Sentiments**
  
![image](https://github.com/user-attachments/assets/edefc471-5948-4d59-a4c8-c8ac10deb381)


### Purpose
- To understand prevalent themes and phrases in the dataset.
- To uncover words heavily influencing sentiment.

---

## Emotion Distribution
Analyzed the distribution of **emotions** in the dataset:
- **Positive**
- **Negative**
- **Neutral**

  ![image](https://github.com/user-attachments/assets/10a6c08a-42bd-4ef9-81fe-1c72f6af35a8)


### Key Findings
- Visualized the emotional landscape of the data.
- Quantified sentiment trends for comparative analysis.

# Modeling

## Unsupervised Learning
### Techniques Tested:
1. **K-Means Clustering**

![image](https://github.com/user-attachments/assets/a357474a-38b3-47aa-8f28-4b93a275dc95)

**KMeans Analysis Summary**

The clustering results indicate poor performance due to the following metrics:

- **High Inertia**: Suggests poorly formed clusters, indicating the need for adjustments like changing the number of clusters.
- **Low Silhouette Score** (~0): Points to poorly defined clusters with overlapping boundaries. Ideally, the score should be above 0.5.
- **High Davies-Bouldin Index** (6.77): Indicates significant overlap between clusters, reflecting poor separation.
- **Negative Adjusted Rand Index (ARI)** (-0.069): Demonstrates that the clustering is worse than random, failing to capture meaningful patterns.

Given the overlap in emotional tones within the dataset, these metrics highlight that the clustering model is unsuitable. Further tuning is unlikely to improve performance. As a next step, alternative unsupervised learning techniques, such as hierarchical agglomerative clustering, will be explored for better results.


2. **Hierarchical Agglomerative Clustering**

   ![image](https://github.com/user-attachments/assets/51e0db88-297a-48fb-a1a3-d8b7fc046c08)


### HAC Performance Summary

The clustering model demonstrates poor performance based on the following metrics:

- **Silhouette Score** (0.2344): Indicates poor clustering with sparse data points and unclear cluster boundaries.
- **Davies-Bouldin Index (DBI)** (1.2317): A relatively high score, suggesting significant overlap between clusters. A DBI below 1 typically reflects well-separated clusters.
- **Normalized Mutual Information (NMI)** (0.0058): Extremely low, showing weak alignment between true labels and predicted clusters, meaning the model fails to capture meaningful patterns.

These metrics highlight the ineffectiveness of unsupervised learning for the dataset. As a result, we conclude that supervised learning techniques, such as decision trees and logistic regression, will be more suitable for analyzing the data.

---

## Supervised Learning
### Models and Performance:

1. **Multinomial Naive Bayes**
   - **Accuracy**: 85%
   - **Challenge**: Struggled with recall for the **negative class**.

2. **Random Forest**
   - Improved **recall** for negative tweets.
   - Faced **overfitting** issues, limiting generalization.

3. **Logistic Regression**
   - **Accuracy**: 74%
   - Delivered **balanced precision and recall** for positive tweets.
   - Overall, the **best-performing model** in terms of robustness and consistency.

# Results

## Best Model
- **Logistic Regression**: Achieved the highest accuracy and demonstrated robust generalization.

---

## Key Insights
### Apple
- **Positive Words**: "great," "love," "awesome," "smart."
- **Negative Words**: "battery life," "autocorrect," "design headache."

### Google
- **Positive Words**: "great," "awesome," "thank," "new."
- **Negative Words**: "network," "product," "lost way."

---

## Recommendations
1. **Collect More Data**  
   - Increase the dataset size, particularly for **underrepresented negative tweets**.

2. **Refine Labeling**  
   - Establish **clear guidelines** for labeling complex cases like sarcasm or mixed sentiments.

3. **Contextual Analysis**  
   - Leverage advanced NLP models (e.g., **BERT**) to enhance sentiment classification accuracy.

4. **Consensus Labeling**  
   - Use **multiple annotators** to minimize bias and improve labeling consistency.


