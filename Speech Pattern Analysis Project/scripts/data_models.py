# Copyright (c) 2025 Rhodri Meredith & Gustav Engelmann
# Licensed under the MIT License. See LICENSE file for details.

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.linear_model import lasso_path
from sklearn.linear_model import Lasso
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def perform_anova(delta_values, delta_name, clusters, unique_clusters):
    
    grouped_data = [delta_values[clusters == cluster] for cluster in unique_clusters]
    f_stat, p_value = stats.f_oneway(*grouped_data)
    
    print(f"ANOVA for {delta_name}: F-statistic = {f_stat:.3f}, p-value = {p_value:.5f}")

def standardise(array):
    return (array - np.mean(array, axis=0))/np.std(array, axis=0)

# Model 1: Linear Regression 
def linear_regression():
    training_size = 25
    np.random.seed(8)
    features = ["Turns durations", "Overlap", "Backchannels by speaker", "Backchannels to speaker", "Skewness", "Decision score"]
    with open(os.path.join(project_root, "output", "model_features_data_method3.csv"), "r", encoding="utf-8") as f:
        data = np.array(list(csv.reader(f))[1:])

        # Seperate into train and test sets and standardise all sets

        np.random.shuffle(data)
        x_train = standardise(data[:training_size, 1:-1].astype(dtype=float))
        x_test = standardise(data[training_size:, 1:-1].astype(dtype=float))
        y_train = standardise(data[:training_size, -1].astype(dtype=float))
        y_test = standardise(data[training_size:, -1].astype(dtype=float))

    alphas, coefs, _ = lasso_path(x_train, y_train, alphas=np.logspace(-4, 2, 100))

    # Compute Train and Test MSE for different alpha values
    train_mse = []
    test_mse = []

    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(x_train, y_train)
        train_mse.append(mean_squared_error(y_train, lasso.predict(x_train)))
        test_mse.append(mean_squared_error(y_test, lasso.predict(x_test)))

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot Lasso Coefficients
    ax1.set_xscale("log")
    ax1.set_xlabel("L1 penalty (lambda)")
    ax1.set_ylabel("Coefficients")
    ax1.set_title("Regularisation Path & Train/Test MSE")

    for i in range(coefs.shape[0]):
        ax1.plot(alphas, coefs[i, :], label=features[i], color=f"C{i}")

    ax1.tick_params(axis='y')
    ax1.legend(loc="upper left", fontsize=8)

    # Create a second y-axis for MSE
    ax2 = ax1.twinx()
    ax2.plot(alphas, train_mse, label="Train MSE", color='black', linestyle='dashed')
    ax2.plot(alphas, test_mse, label="Test MSE", color='red', linestyle='dashed')
    ax2.set_ylabel("Mean Squared Error", color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    ax2.legend(loc="upper right", fontsize=8)

    plt.show()

# Model 2: Random forest classification
def run_random_forest():
    training_size = 25
    testing_size = 30 - training_size
    np.random.seed(3)
    with open(os.path.join(project_root, "output", "model_features_data_method4.csv"), "r", encoding="utf-8") as f:
        data = np.array(list(csv.reader(f))[1:])

        # Seperate into train and test sets and standardise all sets
        np.random.shuffle(data)
        x_train = standardise(data[:training_size, 1:-1].astype(dtype=float))
        x_test = standardise(data[training_size:, 1:-1].astype(dtype=float))
        y_train = data[:training_size, -1].astype(dtype=str)
        y_test = data[training_size:, -1].astype(dtype=str)

    model = RandomForestClassifier(random_state=69).fit(x_train, y_train)
    test_preds = model.predict(x_test)
    print("Classification Report:\n", classification_report(y_test, test_preds))


# Model 3: K-means clustering
def unsupervised_learning(check_optimal_k=False, get_cluster_means=False, get_boxplot=False, get_anova=False, check_norm=False):
    np.random.seed(10)
    training_size = 25
    testing_size = 30 - training_size
    with open(os.path.join(project_root, "output", "model_features_data_method1.csv"), "r", encoding="utf-8") as f:
        data = np.array(list(csv.reader(f))[1:])

        # Seperate into train and test sets and standardise all sets
        np.random.shuffle(data)
        xs = standardise(data[:,1:-1].astype(dtype=float))
        delta_1s = data[:, -1].astype(dtype=float)

    if check_optimal_k:
        inertia = []
        K_range = range(1, 16) 

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=10, n_init=10)
            kmeans.fit(xs)
            inertia.append(kmeans.inertia_)

        # Plot the Elbow Curve
        plt.figure(figsize=(8,5))
        plt.plot(K_range, inertia, marker='o', linestyle='--')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
        plt.title('Elbow Method for Optimal k')
        plt.show()
        
    kmeans = KMeans(n_clusters=10, random_state=10, n_init=10)
    clusters = kmeans.fit_predict(xs)
    unique_clusters = np.unique(clusters)

    with open(os.path.join(project_root, "output", "model_features_data_method2.csv"), "r", encoding="utf-8") as f:
        data = np.array(list(csv.reader(f))[1:])
        delta_2s = data[:, -1].astype(dtype=float)
    
    with open(os.path.join(project_root, "output", "model_features_data_method3.csv"), "r", encoding="utf-8") as f:
        data = np.array(list(csv.reader(f))[1:])
        delta_3s =data[:, -1].astype(dtype=float)

    if get_cluster_means:
        cluster_means = np.array([
            [cluster, np.std(delta_3s[clusters == cluster])]
            for cluster in unique_clusters
        ])
        print(cluster_means)

    if check_norm:
        delta_scores = [delta_1s, delta_2s, delta_3s]
        delta_names = ["Delta1", "Delta2", "Delta3"]

        # Create Q-Q plot for the 3 deltas
        plt.figure(figsize=(12, 8))

        for i, delta in enumerate(delta_scores):
            plt.subplot(1, 3, i + 1)  # 1 row, 3 columns
            stats.probplot(delta, dist="norm", plot=plt)
            plt.title(f"{delta_names[i]} Q-Q Plot")

        plt.tight_layout()
        plt.show()

    if get_anova:
        perform_anova(delta_1s, "Delta1", clusters, unique_clusters)
        perform_anova(delta_2s, "Delta2", clusters, unique_clusters)
        perform_anova(delta_3s, "Delta3", clusters, unique_clusters)

    if get_boxplot:

        # Prepare data for boxplot: List of delta values grouped by cluster
        delta1_by_cluster = [delta_1s[clusters == cluster] for cluster in unique_clusters]
        delta2_by_cluster = [delta_2s[clusters == cluster] for cluster in unique_clusters]
        delta3_by_cluster = [delta_3s[clusters == cluster] for cluster in unique_clusters]

        # Create boxplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

        axes[0].boxplot(delta1_by_cluster, labels=unique_clusters)
        axes[0].set_title("Delta1 by Cluster")
        axes[0].set_xlabel("Cluster")
        axes[0].set_ylabel("Delta Value")

        axes[1].boxplot(delta2_by_cluster, labels=unique_clusters)
        axes[1].set_title("Delta2 by Cluster")
        axes[1].set_xlabel("Cluster")

        axes[2].boxplot(delta3_by_cluster, labels=unique_clusters)
        axes[2].set_title("Delta3 by Cluster")
        axes[2].set_xlabel("Cluster")

        plt.tight_layout()
        plt.show()
