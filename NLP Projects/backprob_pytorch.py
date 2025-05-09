import numpy as np
from collections import Counter, defaultdict
import math
import statistics
import sys
import csv

import torch
import torch.nn as nn

seed = 69
torch.manual_seed(seed)


def read_essay_sets(filename: str) -> dict[int, list[dict]]:
    """Read all essay sets from an ASAP data file.

    Args:
        filename -- tab-separated ASAP data file, cleaned to be unicode
                    compatible

    Returns:
        dict mapping essay set number (int) to a list of dicts, representing
        all the essays in the given essay set.
    """
    essay_set = defaultdict(list)
    total_scores = defaultdict(list)
    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            total_scores[int(row["essay_set"])].append(int(row["domain1_score"]))

    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")        
        for row in reader:
            sd = statistics.stdev(total_scores[int(row["essay_set"])])      
            essay_set[int(row["essay_set"])].append(
                    {"text": row["essay"],
                     "score": (int(row["domain1_score"]) - (sum(total_scores[int(row["essay_set"])]) / len(total_scores[int(row["essay_set"])]))) / sd})
    return essay_set

def extract_features(text: str) -> list[str]:
    """Extract features from a text.

    Extract essay length and a measure of lexical diversity, OVIX.

    Args:
        text -- essay text

    Returns:
        list of feature names
    """
    n = len(text.lower().split())
    k = len(set(text.lower().split()))
    
    essay_length = n ** (1/4)
    ovix = math.log(n) / (2 - (math.log(k)/math.log(n)))
    
    return essay_length, ovix

def construct_matrix(essays: list[dict]) -> tuple:
    """Construct a feature matrix and the target vector

    Args:
        essays: dicts where the keys 'text' and 'score' map to a str and int,
                respectively, representing an essay text and its assigned
                score

    Returns:
        tuple of (X, y, features, feature_idx) where:
        X is a (n_essays, n_features) feature matrix,
        y is a (n_essays) target vector of essay scores,
        features is a list of str of length n_features
        feature_idx is a dict mapping feature strings to their index in
            features
    """
    essay_features = [extract_features(essay["text"])
                      for essay in essays]

    x1 = np.zeros(len(essay_features))
    x2 = np.zeros(len(essay_features))
    
    
    for i in range(len(essay_features)):
        x1[i] = essay_features[i][0]
        x2[i] = essay_features[i][1]
    
    X = np.column_stack((x1, x2))
    
    # The target values are dense, we use the essay scores directly
    y = np.array([essay["score"] for essay in essays], dtype=np.float64)
    return X , y


def without_batches(filepath, epochs: int, learning_rate: float):
    """Trains and evaluates the model

    Args:
        filepath: the path to the data used for training and testing
        epochs: the number of times the model runs in training
        learning rate

    Returns:
        
    """
    device = "cpu"
    # Read the ASAP data (all sets)
    essay_sets = read_essay_sets(filepath)
    # Construct matrices for train set 
    X, y = construct_matrix(essay_sets[1])
    # Construct matrices for test set 
    X2, y2 = construct_matrix(essay_sets[2])
    train_data = torch.tensor(X.astype(np.float32)).to(device)
    train_score = torch.tensor(y.astype(np.float32)).to(device)
    test_data = torch.tensor(X2.astype(np.float32)).to(device)
    test_score = torch.tensor(y2.astype(np.float32)).to(device)
    
    model = torch.nn.Linear(2, 1, bias=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate))
    mse_loss = nn.MSELoss()

    for epoch in range(int(epochs)):
        for i, ex in enumerate(train_data):
            x = ex.unsqueeze(0)
            y = train_score[i].unsqueeze(0)

            model.zero_grad()
            y_pred = model(x).flatten()
            loss = mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()

        y_pred_train = model(train_data).flatten()
        mse_train = mse_loss(y_pred_train, train_score)
        
        y_pred_test = model(test_data).flatten()
        mse_test = mse_loss(y_pred_test, test_score)



        print("MSE Train:", mse_train.item(),"MSE Test:", mse_test.item(), "\n")
    print("Epoch:", epoch + 1,"| Weights:", model.weight.data.detach().numpy(), "| Bias:", model.bias.data.detach().numpy(), "| Learning Rate:", optimizer.param_groups[0]["lr"])
    return ("MSE Train:", mse_train.item(),"MSE Test:", mse_test.item())

def with_batches(filepath, epochs, learning_rate, batch_size):
    batch_size = int(batch_size)
    device = "cpu"
    # Read the ASAP data (all sets)
    essay_sets = read_essay_sets(filepath)
    # Construct matrices for train set 
    X, y = construct_matrix(essay_sets[1])
    # Construct matrices for test set 
    X2, y2 = construct_matrix(essay_sets[2])
    train_data = torch.tensor(X.astype(np.float32)).to(device)
    train_score = torch.tensor(y.astype(np.float32)).to(device)
    test_data = torch.tensor(X2.astype(np.float32)).to(device)
    test_score = torch.tensor(y2.astype(np.float32)).to(device)
    
    model = torch.nn.Linear(2, 1, bias=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate))
    mse_loss = nn.MSELoss()

    for epoch in range(int(epochs)):
        for i in range(0, len(X), batch_size):
            x = train_data[i:i + batch_size]
            y = train_score[i:i + batch_size]

            model.zero_grad()
            y_pred = model(x).flatten()
            loss = mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()

        y_pred_train = model(train_data).flatten()
        mse_train = mse_loss(y_pred_train, train_score)
        
        y_pred_test = model(test_data).flatten()
        mse_test = mse_loss(y_pred_test, test_score)


        print("Epoch:", epoch + 1,"|Weights:", model.weight.data.detach().numpy(), "Bias:|", model.bias.data.detach().numpy(), "Learning Rate:", optimizer.param_groups[0]["lr"], "Batch size:", batch_size)
        print("MSE Train:", mse_train.item(),"MSE Test:", mse_test.item(), "\n")
            
#This solution is rather inelegant, but it works. First two chunks for unbatched, bottom two for batched. 

# def main(filepath, epochs, lr):
#     result = without_batches(filepath, epochs, lr)
#     print(result)
#     return result
    
# if __name__ == '__main__': # To run the code without a batch size, use this main function call with the following arguments [1] training data path, [2] epochs, [3] learning rate
#     main(sys.argv[1], sys.argv[2], sys.argv[3]) 
    
def main(filepath, epochs, lr, batch_size):
    result = with_batches(filepath, epochs, lr, batch_size)
    return result
    
if __name__ == '__main__': # To run the code with a certain batch size, use this main function call with the following arguments [1] training data path, [2] epochs, [3] learning rate, [4] batch size
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])