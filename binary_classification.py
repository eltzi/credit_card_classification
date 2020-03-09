from data_cleaning import education_encoding
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


def split_train_test_set():
    clean_data = education_encoding()
    X_train, X_test, y_train, y_test = train_test_split(clean_data['EDUCATION'].values.reshape(-1,1),
                                                        clean_data['default payment next month'].values,
                                                        test_size=0.2, random_state=24)
    print('mean y train:', np.mean(y_train), 'mean y test:', np.mean(y_test),
          'The positive class fraction in the trainning and testing data are about 22%. This is good, '
          'as these are the same as the overall data, and we can say that the training set is representative '
          'of the testing set')
    return  X_train, X_test, y_train, y_test

def classifier():
    X_train, X_test, y_train, y_test = split_train_test_set()

    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    print('% of correct classifications:', lr.score(X_test, y_test))
    print(metrics.confusion_matrix(y_test, y_pred))
    return lr

def predicted_probabilities():
    model = classifier()
    X_train, X_test, y_train, y_test = split_train_test_set()
    print(np.sum(model.predict_proba(X_test),1))
    print(np.unique(np.sum(model.predict_proba(X_test),1)))
    positive_propability = (model.predict_proba(X_test))[:,1]
    print('probability of membership in positive class', positive_propability)
    plt.hist(positive_propability)
    plt.xlabel('Predicted probability of positive class for testing data')
    plt.ylabel('Number of samples')
    plt.show()

    pos_sample_pos_prob = positive_propability[y_test==1]
    neg_sample_pos_prob = positive_propability[y_test==0]

    plt.hist([pos_sample_pos_prob, neg_sample_pos_prob], histtype='barstacked')
    plt.legend(['Positive Samples', 'Negative Samples'])
    plt.xlabel('Predicted probability of positive class')
    plt.ylabel('Number of samples')
    plt.show()

def accuracy_metrics():
    """
    # calculate accuracy by creating a logical mask that is True whenever the predicted label is equal to the actual
    label, and False otherwise. We can then take the average of the mask, giving us the proportion of the correct
    classification
    """

    X_train, X_test, y_train, y_test = split_train_test_set()
    y_pred = classifier()

    is_correct = y_pred == y_test
    print(np.mean(is_correct))
    print('accuracy score', metrics.accuracy_score(y_test, y_pred))


def random_data_classifier():
    X = np.random.uniform(low=0.0, high=10.0, size=(1000,))
    slope = 0.25
    intercept = -1.25
    y = slope * X + np.random.normal(loc=0.0, scale=1.0, size=(1000,)) + intercept

    lin_reg = LinearRegression()
    lin_reg.fit(X.reshape(-1,1), y)
    print(lin_reg.intercept_, lin_reg.coef_)
    y_pred = lin_reg.predict(X.reshape(-1,1))
    plt.scatter(X, y, s=1)
    plt.plot(X, y_pred, 'r')
    plt.show()

