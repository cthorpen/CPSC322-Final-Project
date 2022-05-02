import utils
import matplotlib.pyplot as plt

def bar_chart(x, y, title, xlabel, ylabel, addLabels=False, labels=None):
    plt.figure(figsize=(20,5))
    plt.bar(x, y)
    if addLabels == True:
        addtext(x,labels)
    plt.xlabel(xlabel, fontsize=17, color='r')
    plt.ylabel(ylabel, fontsize=17, color='r')
    plt.title(title, fontsize=20, color='r')
    plt.show()

def pie_chart(x, labels, title):
    plt.figure(figsize=(20,5))
    plt.pie(x, labels=labels, autopct="%1.1f%%")
    plt.title(title, fontsize=20, color='r')

def addtext(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])

def histogram(data, title, xlabel, ylabel):
    plt.figure(figsize=(20,5))
    n, bins, patches = plt.hist(data, bins=(10), edgecolor='black')
    plt.xticks(bins)
    plt.xlabel(xlabel, fontsize=17, color='r')
    plt.ylabel(ylabel, fontsize=17, color='r')
    plt.title(title, fontsize=20, color='r')
    plt.show()

def scatter_chart(x, y, title, xlabel, ylabel):
    plt.figure()
    plt.scatter(x, y, s=100, c="purple")
    m, b = utils.compute_slope_intercept(x, y)
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="r", lw=5)
    m = round(m, 4)
    b = round(b, 4)
    print('y =', m, 'x +', b)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    r = utils.compute_correlation_coefficient(x, y)
    print('r:', round(r, 2))
    cov = utils.compute_covariance(x, y)
    print('Covariance:', round(cov, 2))
    plt.show()

def box_plot(distributions, labels, title, xlabel, ylabel): # distributions and labels are parallel
    # distributions: list of 1D lists of values
    plt.figure(figsize=(20,5))
    plt.boxplot(distributions)
    plt.xticks(list(range(1, len(distributions) + 1)), labels, rotation=45)
    plt.xlabel(xlabel, fontsize=17, color='r')
    plt.ylabel(ylabel, fontsize=17, color='r')
    plt.title(title, fontsize=20, color='r')
    plt.show()