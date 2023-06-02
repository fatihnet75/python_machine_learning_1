
import h5py
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

file = h5py.File('C:/Users/fatih/OneDrive/Masaüstü/veri bilimi ödev/Features_Frequency_Alpha_Valence.mat', 'r')

matlab_data = file['Features_Frequency_Alpha_Valence']

X = matlab_data[:-1, :]
y = matlab_data[-1, :].flatten()  


X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.2, random_state=42)

naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
naive_bayes_pred = naive_bayes.predict(X_test)
naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_pred) * 100

print("Naive Bayes Başarı: %.2f%%" % naive_bayes_accuracy)

file.close()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.2, random_state=42)

# LDA
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)
lda_pred = lda_model.predict(X_test)
lda_accuracy = accuracy_score(y_test, lda_pred) * 100

print("Linear Discriminant Analysis (LDA) Accuracy: %.2f%%" % lda_accuracy)

file.close()



from sklearn.svm import SVC


X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.2, random_state=42)

svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred) * 100

print("SVM Accuracy: %.2f%%" % svm_accuracy)

file.close()
from sklearn.neighbors import KNeighborsClassifier


X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred) * 100

print("kNN Accuracy: %.2f%%" % knn_accuracy)

file.close()


from sklearn.ensemble import RandomForestClassifier



X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.2, random_state=42)

random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
random_forest_pred = random_forest.predict(X_test)
random_forest_accuracy = accuracy_score(y_test, random_forest_pred) * 100

print("Random Forest Accuracy: %.2f%%" % random_forest_accuracy)

file.close()



