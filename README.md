# Movie Recommendation System using Google Colab

## Overview

The Movie Recommendation System is implemented in Google Colab, providing personalized movie suggestions based on user input. It employs various recommendation techniques such as content-based filtering and collaborative filtering to recommend movies similar to those a user likes. The system processes movie data from a CSV dataset and incorporates textual features like genre, keywords, and cast using TF-IDF vectorization. Cosine similarity is used to compute similarity scores between movies, enabling accurate recommendations.

## Google Colab Setup

To run the Movie Recommendation System in Google Colab, follow these steps:

1. **Open Google Colab:** Visit [Google Colab](https://colab.research.google.com/).

2. **Upload your notebook:** Upload your notebook file (`recommendation_system.ipynb`) containing the implementation.

3. **Connect to Google Drive (if necessary):** If your dataset or files are in Google Drive, mount your Google Drive in Colab.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Install dependencies (if necessary): If additional libraries are required, install them using pip.

```python
!pip install pandas scikit-learn
```

## Features

### Data Import and Preprocessing:
```python
import pandas as pd

# Load dataset
df = pd.read_csv("Movies Recommendation.csv")

# Process data
df_features = df[["Movie_Genre", "Movie_Keywords", "Movie_Cast", "Movie_Director"]].fillna("")
```

### Feature Engineering:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(df_features["Movie_Genre"] + " " + df_features["Movie_Keywords"] + " " + df_features["Movie_Cast"] + " " + df_features["Movie_Director"])

```
### Recommendation Generation:
```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity scores
similarity_scores = cosine_similarity(x)

# Recommend top movies
top_movies_indices = similarity_scores[0].argsort()[:-11:-1]
```
### User Interaction:
```python
import difflib

# User input and validation
favorite_movie = input("Enter your favorite movie: ")
closest_matches = difflib.get_close_matches(favorite_movie, df['Movie_Title'], n=1)
```
## Dataset
```
https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Movies%20Recommendation.csv
```

## Usage
### Input: Enter your favorite movie title when prompted.
### Output: Receive a list of top recommended movies based on similarity scores.

## Evaluation
The system's performance can be evaluated using metrics like precision, recall, and F1-score to assess recommendation accuracy against user preferences.

## Credits
@Developed by [https://github.com/yogeshgupta100]

## License
This project is licensed under the MIT License - see the LICENSE file for details.




