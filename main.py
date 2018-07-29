import util
import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
from colorama import init
from termcolor import colored
import os
import glob


def main():
    init()

    # get the dataset
    mypath = 'dataset/'
    # remove any newlines or spaces at the end of the input
    path = mypath.strip('\n')
    if path.endswith(' '):
        path = path.rstrip(' ')

    # preprocess data into two folders instead of 6
    reorganize_dataset(path)

    # do the main test
    main_test(path)


def reorganize_dataset(path):
    likes = ['comp.graphics', 'sci.space', 'talk.religion.misc']
    dislikes = ['comp.windows.x', 'rec.motorcycles', 'misc.forsale']

    folders = glob.glob(os.path.join(path, '*'))
    if len(folders) == 2:
        return
    else:
        # create `likes` and `dislikes` directories
        if not os.path.exists(os.path.join(path, 'likes')):
            os.makedirs(os.path.join(path, 'likes'))
        if not os.path.exists(os.path.join(path, 'dislikes')):
            os.makedirs(os.path.join(path, 'dislikes'))

        for like in likes:
            files = glob.glob(os.path.join(path, like, '*'))
            for f in files:
                parts = f.split(os.sep)
                name = parts[len(parts) - 1]
                newname = like + '_' + name
                os.rename(f, os.path.join(path, 'likes', newname))
            os.rmdir(os.path.join(path, like))

        for like in dislikes:
            files = glob.glob(os.path.join(path, like, '*'))
            for f in files:
                parts = f.split(os.sep)
                name = parts[len(parts) - 1]
                newname = like + '_' + name
                os.rename(f, os.path.join(path, 'dislikes', newname))
            os.rmdir(os.path.join(path, like))



def remove_incompatible_files(dir_path):
    # find incompatible files
    print colored('Finding files incompatible with utf8: ', 'green', attrs=['bold'])
    incompatible_files = util.find_incompatible_files(dir_path)
    print colored(len(incompatible_files), 'yellow'), 'files found'

    # delete them
    if(len(incompatible_files) > 0):
        print colored('Deleting incompatible files', 'red', attrs=['bold'])
        util.delete_incompatible_files(incompatible_files)



def main_test(path=None):
    dir_path = path

    remove_incompatible_files(dir_path)

    print '\n\n'

    # load data
    print colored('Loading files into memory', 'green', attrs=['bold'])
    files = sklearn.datasets.load_files(dir_path)

    # refine all refine_all_emails
    print colored('Refining all files', 'green', attrs=['bold'])
    util.refine_all_emails(files.data)

    # calculate the BOW representation
    print colored('Calculating BOW', 'green', attrs=['bold'])
    word_counts = util.bagOfWords(files.data)

    # TFIDF
    print colored('Calculating TFIDF', 'green', attrs=['bold'])
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
    X = tf_transformer.transform(word_counts)

    print '\n\n'
    
    # defining test_size
    test_size = [0.2]

    # create classifier
    print colored('TFIDF with Naive Bayes', 'red', attrs=['bold'])
    clf = sklearn.naive_bayes.MultinomialNB()

    # print '\n'
    for test in test_size:
        test_classifier(X, files.target, clf, test, y_names=files.target_names, confusion=False)


    print '\n\n'

    print colored('TFIDF with Support Vector Machine', 'red', attrs=['bold'])
    clf = sklearn.svm.LinearSVC()

    # print '\n'
    for test in test_size:
        test_classifier(X, files.target, clf, test, y_names=files.target_names, confusion=False)


    print '\n\n'
    
    print colored('TFIDF with K-Nearest Neighbours', 'red', attrs=['bold'])
    n_neighbors = 11
    weights = 'uniform'
    weights = 'distance'
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

    # test the classifier
    # print '\n'
    for test in test_size:
        test_classifier(X, files.target, clf, test, y_names=files.target_names, confusion=False)



def test_classifier(X, y, clf, test_size, y_names=None, confusion=False):
    # train-test split
    print 'test size is: %2.0f%%' % (test_size * 100)
    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=test_size)

    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    if not confusion:
        print colored('Classification report:', 'magenta', attrs=['bold'])
        print sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names)
    else:
        print colored('Confusion Matrix:', 'magenta', attrs=['bold'])
        print sklearn.metrics.confusion_matrix(y_test, y_predicted)

if __name__ == '__main__':
    main()
