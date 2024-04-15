# Emotion-Analysis-Chatbot

# I.  Introduction:

#### Project Overview:
The initial development phase of the code base involves implementing a model for emotion classification based on given dataset:

a) Baseline Emotion Prediction Model with Classic Classifiers:

  - Data Preprocessing: Cleaning and preprocessing data from provided datasets.
  - Feature Engineering: Extracting relevant features for classic classifiers.
  - Model Selection: Choosing classic classifiers like Logistic Regression, Naive Bayes, etc.
  - Model Training: Training classifiers and evaluating performance metrics.
    
b) Data Analysis:

 You are given a sample of real-time bidding data that correspond with
our programmatic advertising solution. You are required to perform data
analysis, identify and present any insights you are able to derive from the data,
following the method of your choice. The schema explanation is given in
“schema_explained.txt”.


#### Dataset:
Each entry in this dataset consists of a text segment representing a Twitter message and a corresponding label indicating the predominant emotion conveyed. The emotions are classified into six categories: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5). Whether you're interested in sentiment analysis, emotion classification, or text mining, this dataset provides a rich foundation for exploring the nuanced emotional landscape within the realm of social media.

- data.csv : This file includes the data for our models

You can find the dataset [here](https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data).


### Installation

    # go to your home dir
    git clone https://github.com/jvario/emotions-analysis-chatbot.git


  #### api - calls
 - **/emotions_clf/LoadEmotionsModel?model_name=SGD** :  loading a model and and run pretrained LLM
 - **/emotionc_clf/TrainEmotionsModel** : training emotion models
 - 
# II.  Pipeline:

#### Preproccess:
In order to perform text sanitization on our data, we applied the following steps:

- Remove Panctuation
- Remove StopWords
- Tokenization
- Lemmatization

#### Baseline Emotions Prediction Model with Classic Classifiers:
In our single-label classification problem, where each data point is associated with only one tag, we carefully divided the dataset into training and testing sets. This division is crucial for evaluating the performance of our models effectively. By separating the data into distinct training and testing subsets, we ensure that our models are trained on one portion of the data and evaluated on another, allowing us to gauge their generalization capability accurately. 

Additionally, for feature extraction, we've applied **TF-IDF (Term Frequency-Inverse Document Frequency)**. This technique helps to represent each document in the dataset as a vector based on the importance of each word, considering both its frequency in the document and its rarity across all documents.

Furthermore, we've utilized **Label Binarization** to encode the single labels into binary format, facilitating the classification task.
For evaluating the performance of our models, we've chosen several metrics including **recall, F1 score, support, Jaccard score, and precision**. These metrics provide insights into different aspects of the model's performance, such as its ability to correctly classify each label, handle imbalanced data, and capture the trade-off between precision and recall.


# III.  Results:

| Emotion Model | Sample Size | Accuracy | Jaccard Score |
|-------|-------------|----------|---------------|
| SVM   | ~416810      | 0.78     | 0.55          |
| SGD   | ~416810      | 0.78     | 0.56          |

This table represents the evaluation results for different models based on dataset. The metrics include Accuracy, Jaccard score.

Pretrained LLM used for augmented generation was [Gemma](https://huggingface.co/blog/gemma).

# IV. Conclusion:
Both the SVM and SGD models have similar accuracy scores of 0.68, indicating that they correctly predicted around 68% of the samples. This level of accuracy is quite commendable, considering the time limitation and dataset constraints. Additionally, when considering the Jaccard Score, which measures the similarity between two sets, the SGD model slightly outperforms the SVM model with a score of 0.56 compared to 0.55.

In conclusion, both models demonstrate relatively good performance within the given limitations, with the SGD model showing a slightly better Jaccard Score. Further analysis, including examining additional performance metrics and conducting cross-validation, would provide a more comprehensive evaluation of the models' effectiveness.
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
