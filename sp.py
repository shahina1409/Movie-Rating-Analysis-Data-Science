import pandas as pd

# Load Datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Dataset Shapes
print("Movies Dataset Shape:", movies.shape)
print("Ratings Dataset Shape:", ratings.shape)

# First 5 Rows
print("\nMovies Dataset:")
print(movies.head())

print("\nRatings Dataset:")
print(ratings.head())

# Dataset Information
print("Movies Dataset Info:")
print(movies.info())

print("\nRatings Dataset Info:")
print(ratings.info())

# Statistical Summary
print("\nRatings Dataset Statistics:")
print(ratings.describe())

# Missing Values Check
print("\nMissing Values in Movies Dataset:")
print(movies.isnull().sum())

print("\nMissing Values in Ratings Dataset:")
print(ratings.isnull().sum())

# =====================
# STEP 5: DATA CLEANING
# =====================

# Check Duplicate Records
print("Duplicate Rows in Movies Dataset:", movies.duplicated().sum())
print("Duplicate Rows in Ratings Dataset:", ratings.duplicated().sum())

# Remove Duplicates
movies = movies.drop_duplicates()
ratings = ratings.drop_duplicates()

# Check Missing Values Again
print("\nMissing Values in Movies Dataset:")
print(movies.isnull().sum())

print("\nMissing Values in Ratings Dataset:")
print(ratings.isnull().sum())

# Final Shape After Cleaning
print("\nMovies Dataset Shape After Cleaning:", movies.shape)
print("Ratings Dataset Shape After Cleaning:", ratings.shape)

# =====================
# STEP 6: RATING DISTRIBUTION ANALYSIS
# =====================

import matplotlib.pyplot as plt

print("Average Rating:", ratings['rating'].mean())
print("Highest Rating:", ratings['rating'].max())
print("Lowest Rating:", ratings['rating'].min())

# Histogram
plt.figure(figsize=(8,5))
plt.hist(ratings['rating'], bins=10)
plt.title("Movie Rating Distribution")
plt.xlabel("Ratings")
plt.ylabel("Frequency")
plt.show()

# =====================
# STEP 7: TOP 10 HIGHEST RATED MOVIES
# =====================

# Merge datasets
movie_ratings = pd.merge(ratings, movies, on='movieId')

# Average rating per movie
top_movies = movie_ratings.groupby('title')['rating'].mean()

# Top 10 movies
top_10 = top_movies.sort_values(ascending=False).head(10)

print("Top 10 Highest Rated Movies:")
print(top_10)

# Visualization
plt.figure(figsize=(10,6))
top_10.plot(kind='bar')

plt.title("Top 10 Highest Rated Movies")
plt.xlabel("Movie Title")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =====================
# STEP 8: TOP 10 MOST RATED MOVIES
# =====================

# Count ratings per movie
most_rated = movie_ratings.groupby('title')['rating'].count()

# Top 10 most rated movies
top_10_rated = most_rated.sort_values(ascending=False).head(10)

print("Top 10 Most Rated Movies:")
print(top_10_rated)

# Visualization
plt.figure(figsize=(10,6))
top_10_rated.plot(kind='bar')

plt.title("Top 10 Most Rated Movies")
plt.xlabel("Movie Title")
plt.ylabel("Number of Ratings")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =====================
# STEP 9: GENRE ANALYSIS
# =====================

# Split genres
genres = movies['genres'].str.split('|').explode()

# Count genres
genre_count = genres.value_counts()

print("Genre Counts:")
print(genre_count.head(10))

# Visualization
plt.figure(figsize=(10,6))
genre_count.head(10).plot(kind='bar')

plt.title("Top 10 Movie Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =====================
# STEP 10: YEAR-WISE MOVIE RELEASE TREND
# =====================

# Extract year from title
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')

# Remove missing years
movies_year = movies.dropna(subset=['year'])

# Count movies per year
year_count = movies_year['year'].value_counts().sort_index()

print("Movies Released Per Year:")
print(year_count.tail(10))

# Visualization
plt.figure(figsize=(12,6))
year_count.plot(kind='line')

plt.title("Movies Released Per Year")
plt.xlabel("Year")
plt.ylabel("Number of Movies")
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================
# STEP 11: RATING VS NUMBER OF RATINGS
# =====================

# Average Rating per Movie
avg_rating = movie_ratings.groupby('title')['rating'].mean()

# Number of Ratings per Movie
num_ratings = movie_ratings.groupby('title')['rating'].count()

# Create DataFrame
movie_stats = pd.DataFrame({
    'Average Rating': avg_rating,
    'Number of Ratings': num_ratings
})

print(movie_stats.head())

# Scatter Plot
plt.figure(figsize=(10,6))
plt.scatter(
    movie_stats['Number of Ratings'],
    movie_stats['Average Rating']
)

plt.title("Average Rating vs Number of Ratings")
plt.xlabel("Number of Ratings")
plt.ylabel("Average Rating")
plt.grid(True)
plt.show()

# =====================
# STEP 12: CORRELATION HEATMAP
# =====================

import seaborn as sns

# Correlation Matrix
correlation = ratings[['userId', 'movieId', 'rating']].corr()

print("Correlation Matrix:")
print(correlation)

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(correlation,
            annot=True,
            cmap='coolwarm',
            linewidths=0.5)

plt.title("Correlation Heatmap")
plt.show()

# =====================
# STEP 13: INSIGHTS & CONCLUSION
# =====================

print("\n===== PROJECT INSIGHTS =====\n")

print("1. Dataset contains movie information and user ratings.")
print("2. Most ratings are concentrated between 3 and 4.5.")
print("3. Some movies have very high ratings but very few reviews.")
print("4. Popular movies receive a large number of ratings.")
print("5. Drama, Comedy and Action are among the most common genres.")
print("6. Movie releases increased significantly in recent years.")
print("7. Ratings data provides useful insights into audience preferences.")

print("\n===== CONCLUSION =====\n")

print("This project analyzed movie ratings using Python.")
print("Data cleaning, visualization and exploratory data analysis were performed.")
print("The analysis helped identify popular movies, rating trends and genre distributions.")
print("These insights can be useful for recommendation systems and movie trend analysis.")

movie_stats = movie_ratings.groupby('title').agg({
    'rating':['mean','count']
})

movie_stats.columns=['avg_rating','num_ratings']

popular_movies = movie_stats[movie_stats['num_ratings']>=100]

top_movies = popular_movies.sort_values(
    'avg_rating',
    ascending=False
).head(10)

print(top_movies)

genre_rating = movie_ratings.groupby('genres')['rating'].mean()

genre_rating.sort_values(
    ascending=False
).head(10).plot(kind='bar')
plt.show()

active_users = ratings['userId'].value_counts().head(10)

active_users.plot(kind='bar')
plt.show()

movie_matrix = movie_ratings.pivot_table(
    index='userId',
    columns='title',
    values='rating'
)

corr_matrix = movie_matrix.corr()

movie_name = "Toy Story (1995)"

recommendations = corr_matrix[movie_name].sort_values(
    ascending=False
).head(10)

print(recommendations)