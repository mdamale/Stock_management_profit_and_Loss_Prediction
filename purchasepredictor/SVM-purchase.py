# Import libraries and packages
from sklearn import svm, datasets
import pickle 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.manifold import Isomap


# Load Sample data
#iris = datasets.load_iris()
stock = pd.read_csv('stock.csv', sep=',')

# Split loaded data into independent and target features
X = stock.iloc[:, :-1].values 
y = stock.iloc[:, -1].values 

plt.scatter(X[:, 0], X[:, 2], c=y, s=50, cmap='autumn');
plt.xlabel('Product')
plt.ylabel('Price')
plt.show() #product-price

plt.scatter(X[:, 2], X[:, 3], c=y, s=50, cmap='autumn');
plt.xlabel('Price')
plt.ylabel('Supplier')
plt.show() #price-supplier

plt.scatter(X[:, 3],y, c=y, s=50, cmap='autumn');
plt.ylabel('Accept/Reject')
plt.xlabel('Supplier')
plt.show() #accept-supplier

# Train Support Vector Machine (SVM) model with all data 
svmModel = svm.SVC(kernel='linear').fit(X, y)

# Persist model so that it can be used by different consumers
svmFile = open('SVMModel.pckl', 'wb')
pickle.dump(svmModel, svmFile)
svmFile.close()


# Create an isomap and fit the `digits` data to it
X_iso = Isomap(n_neighbors=10).fit_transform(X)



# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Adjust layout
fig.suptitle('Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots 

ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=y)
ax[0].set_title('Actual Training Labels')

# Show the plots
plt.show()











