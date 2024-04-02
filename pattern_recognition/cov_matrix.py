import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X = np.random.randint(5, 20, size=(3,4))
    X = np.array([[3, 3, 3, 3], 
                  [1, 2, 4, 5], 
                  [1, 4, 2, 5]])
    # X = np.array([[1, 3, 5, 7], 
    #               [1, 2, 3, 4], 
    #               [1, 2, 3, 1]])
    print(X)
    Mu = np.mean(X, axis=1)
    print(Mu)
    X_diff = (X.T - Mu).T
    X_cov = np.dot(X_diff, X_diff.T) / 4
    print(X_cov)
    
    print(X[0])
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    ax1.scatter(X[0], X[1])
    ax2.scatter(X[0], X[2])
    ax3.scatter(X[1], X[2])
    
    plt.show()