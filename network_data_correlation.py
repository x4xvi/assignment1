import csv
import random
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Function to generate network packet data
def generate_data(samples):
    data = []
    for _ in range(samples):
        packet_size = random.randint(100, 2000)
        error_number = random.randint(0, 20)
        duration = round(random.uniform(0.1, 10.0), 2)
        protocol_number = random.choice([1, 2])  # 1 for TCP, 2 for UDP
        data.append([packet_size, error_number, duration, protocol_number])
    return data

# Write features to CSV file
def write_features_to_csv(features, filename):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(features)

# Read features from CSV file
def read_features_from_csv(filename):
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        features = next(csvreader)  # Read the first row (header)
    return features

# Write data to CSV file
def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)

# Read data from CSV file
def read_from_csv(filename):
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        data = list(csvreader)
    return data

# Calculate correlation matrix using Pearson formula
def calculate_correlation_matrix(data):
    df = pd.DataFrame(data).astype(float)
    correlation_matrix = df.corr(method='pearson')
    return correlation_matrix

# Print correlation matrix with feature names as headers
def print_correlation_matrix(correlation_matrix, features):
    print("Correlation Matrix:")
    print(pd.DataFrame(correlation_matrix.values, columns=features, index=features))

# Function to plot and save correlation matrix as PDF
def save_correlation_matrix_as_pdf(correlation_matrix, features, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('Correlation Matrix')
    plt.xticks(np.arange(len(features)), features, rotation=45)
    plt.yticks(np.arange(len(features)), features)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Find names of two features with highest correlation
def find_highest_correlation(correlation_matrix, features):
    max_corr = 0
    feature1 = ''
    feature2 = ''
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if abs(correlation_matrix.iloc[i, j]) > max_corr:
                max_corr = abs(correlation_matrix.iloc[i, j])
                feature1 = features[i]
                feature2 = features[j]
    return feature1, feature2

# Generate and write features to CSV file
features = ['Packet Size', 'Error Number', 'Duration', 'Protocol Number']
write_features_to_csv(features, 'features.csv')

# Generate and write data to CSV file
data = generate_data(200)
write_to_csv(data, 'network_packet_data.csv')

# Read features from CSV file
features = read_features_from_csv('features.csv')

# Read data from CSV file
data_read = read_from_csv('network_packet_data.csv')

# Calculate correlation matrix
correlation_matrix = calculate_correlation_matrix(data_read)

# Save correlation matrix as PDF
save_correlation_matrix_as_pdf(correlation_matrix, features, 'correlation_matrix.pdf')

# Find names of two features with highest correlation
highest_corr_features = find_highest_correlation(correlation_matrix, features)

# Write names of features with highest correlation to PDF
# with open('highest_correlation.pdf', 'w') as f:
#     f.write(f'Two features with highest correlation: {highest_corr_features[0]} and {highest_corr_features[1]}')


# Print correlation matrix with feature names as headers
print_correlation_matrix(correlation_matrix, features)
def create_highest_correlation_pdf(filename, feature1, feature2):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 700, f'Two features with highest correlation: {feature1} and {feature2}')
    c.save()

create_highest_correlation_pdf('highest_correlation.pdf', *highest_corr_features)
