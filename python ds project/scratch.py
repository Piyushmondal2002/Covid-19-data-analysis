import os
from kaggle.api.kaggle_api_extended import KaggleApi
from requests import head

# Setting my Kaggle API key
os.environ["KAGGLE_USERNAME"] = "pijushmondal510"
os.environ["KAGGLE_KEY"] = "7873c9aae8a58f7fe9a6bfe5ffa3ea73"

# Instantiating Kaggle API
api = KaggleApi()
api.authenticate()

# Downloading the dataset
api.dataset_download_files("imdevskp/corona-virus-report", path="./dataset", unzip=True)

# Loading the dataset into a DataFrame
import pandas as pd
df = pd.read_csv("dataset/covid_19_clean_complete.csv")

# Displaying the first 10 rows of the DataFrame
print(df.head(10))

# Grouping and resampling the data

grouped = df.groupby('Country/Region')['Confirmed'].sum()

# Data cleaning and making the data ready for analysis.

# Handling missing value:
# # Handling missing values using forward fill
df.ffill(inplace=True)  # Forward fill missing valuesForward fill missing values
# OR
df.dropna(inplace=True)  # Drop rows with missing values

# Removing duplicates
df.drop_duplicates(inplace=True)

# Data type conversion
df['Date'] = pd.to_datetime(df['Date'])


# Data operations: Finding the mean , median , st. dev and doing statistical analysis:

import numpy as np

# Calculate mean
mean_confirmed = np.mean(df['Confirmed'])
print(mean_confirmed)

# Calculate median
median_confirmed = np.median(df['Confirmed'])
print(median_confirmed)

# Calculate standard deviation
std_confirmed = np.std(df['Confirmed'])
print(std_confirmed)

# Descriptive statistics
print(df.describe())

# Correlation analysis
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()


# Using Matplotlib for Visualization:
import matplotlib.pyplot as plt

# Plotting confirmed cases over time
plt.plot(df['Date'], df['Confirmed'], label='Confirmed')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.title('COVID-19 Confirmed Cases Over Time')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Plotting a histogram of confirmed cases
plt.hist(df['Confirmed'], bins=20, color='blue', alpha=0.7)
plt.xlabel('Confirmed Cases')
plt.ylabel('Frequency')
plt.title('Histogram of COVID-19 Confirmed Cases')
plt.show()

# Plotting a scatter plot of confirmed vs. recovered cases
plt.scatter(df['Confirmed'], df['Recovered'], color='red', alpha=0.5)
plt.xlabel('Confirmed Cases')
plt.ylabel('Recovered Cases')
plt.title('Scatter Plot of COVID-19 Confirmed vs. Recovered Cases')
plt.show()

# Plotting a bar plot of top 10 countries with highest confirmed cases
top_countries = df.groupby('Country/Region')['Confirmed'].max().nlargest(10)
top_countries.plot(kind='bar', color='green', alpha=0.7)
plt.xlabel('Country')
plt.ylabel('Confirmed Cases')
plt.title('Top 10 Countries with Highest Confirmed Cases')
plt.xticks(rotation=45)
plt.show()

# using seaborn library for Data Visualization:

import seaborn as sns

# Plotting heatmap of correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Plotting pairplot for selected variables
sns.pairplot(df[['Confirmed', 'Recovered', 'Deaths']])
plt.title('Pairplot of Confirmed, Recovered, and Deaths')
plt.show()

# Plotting boxplot of confirmed cases
sns.boxplot(x='Country/Region', y='Confirmed', data=df.head(20))
plt.title('Boxplot of Confirmed Cases by Country')
plt.xticks(rotation=45)
plt.show()

# Plotting barplot of top 10 countries with highest confirmed cases
plt.figure(figsize=(10, 6))
top_countries = grouped.nlargest(10)
sns.barplot(x=top_countries.index, y=top_countries.values, palette='viridis')
plt.xlabel('Country')
plt.ylabel('Confirmed Cases')
plt.title('Top 10 Countries with Highest Confirmed Cases')
plt.xticks(rotation=45)
plt.show()










