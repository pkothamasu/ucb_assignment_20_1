###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import fbeta_score, accuracy_score

def distribution(data, transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    # Create figure
    fig = pl.figure(figsize = (11,5));

    # Skewed feature plotting
    for i, feature in enumerate(['capital-gain','capital-loss']):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
  
    # Create figure
    fig, ax = pl.subplots(2, 3, figsize = (11,7))

    # Constants
    bar_width = 0.3
    colors = ['r','b','g','y']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    #pl.tight_layout()
    pl.show()
    
    

def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize = (9,5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    pl.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize = 12)
    pl.xlabel("Feature", fontsize = 12)
    
    pl.legend(loc = 'upper center')
    pl.tight_layout()
    pl.show()  
    
    
#Draw plots with confustion-matrix, ROC curve, Precision-Recall curve
def draw_plots( clf, X_test, y_test, clf_text ):
    """ draw plots with confustion-matrix, ROC curve, and Precision-Recall curve
        returns vectors fpr, tpr, precision, recall
        returns scalar auc
    :param clf: classifier 
    :param X_test: test split features X-independent variables
    :param y_test: test split target labels y-dependent variables
    :param clf_text: string name of classifier for labels 
    """  
    y_test_pred = clf.predict(X_test) 
    y_pred_proba = clf.predict_proba(X_test)[:,1]
    
    #plot the Confusion-Matrix curve in upper-left
    fig, axs = pl.subplots(1,3, figsize=(16,5))
    fig.tight_layout(pad=5)
    clf_confusion_matrix = confusion_matrix( y_test, y_test_pred, normalize='true')
    sns.heatmap( clf_confusion_matrix, annot=True, ax=axs[0] )
    axs[0].set_xlabel('Predicted')
    axs[0].set_ylabel('Truth')
    axs[0].set_title('Confusion Matrix')

    #plot the Precision-Recall Curve in the lower-right
    precision = precision_score( y_test, y_test_pred )
    recall    = recall_score( y_test, y_test_pred )
    f1        = f1_score( y_test, y_test_pred )
    clf_precision, clf_recall, prc_thresholds = precision_recall_curve( y_test, y_pred_proba )
    axs[1].plot( clf_recall, clf_precision, label=clf_text )
    axs[1].set_ylabel('Precision')
    axs[1].set_xlabel('Recall')
    axs[1].set_title('Precision-Recall Curve')
    axs[1].plot( recall, precision, 'o' )
    txtlabel = f'f1={f1:.3f}'
    axs[1].text( recall, precision, txtlabel)
    axs[1].legend()
    
    #plot the ROC curve in upper-right
    clf_fpr, clf_tpr, roc_thresholds = roc_curve( y_test,  y_pred_proba )
    clf_auc = roc_auc_score( y_test, y_pred_proba )
    legend_label_text = f'{clf_text} AUC={str(round(clf_auc,3))}'
    axs[2].plot( clf_fpr, clf_tpr, label=legend_label_text )
    axs[2].set_ylabel('True Positive Rate')
    axs[2].set_xlabel('False Positive Rate')
    axs[2].set_title('ROC')
    axs[2].legend()
    
    pl.show()    
    return clf_fpr, clf_tpr, clf_auc, clf_precision, clf_recall


#Computes all the scores
def evaluate_scores( clf, X_train, y_train, X_test, y_test, clf_text ):
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    y_test_pred = clf.predict( X_test )
    y_pred_proba = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score( y_test, y_pred_proba )
    tn, fp, fn, tp = confusion_matrix( y_test, y_test_pred ).ravel()
    spec = tn/(tn+fp)
    acc  = accuracy_score( y_test, y_test_pred )
    prec = precision_score( y_test, y_test_pred )
    rec  = recall_score( y_test, y_test_pred )
    f1   = f1_score( y_test, y_test_pred )
    if clf_text == 'Baseline':
        fit_time = np.nan
    else:
        fit_time = np.mean( clf.cv_results_['mean_fit_time'])
        
    print(f'acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} spec={spec:.3f} fit_time={fit_time:.3f}')

    #auc = roc_auc_score( y_test, y_pred_proba )
    
    score_dict = {'classifier' : [clf_text], 
                  'train_score':[round(train_score,3)], 
                  'accuracy':[round(acc,3)], 
                  'precision':[round(prec,3)], 
                  'recall':[round(rec,3)], 
                  'f1':[round(f1,3)], 
                  'specificity':[round(spec,3)],
                  'auc':[round(auc,3)],
                  'fit_time':[round(fit_time,3)] }
    df = pd.DataFrame(data=score_dict)
    return df


# Traina and predict on training and test dataset
# Import two metrics from sklearn - fbeta_score and accuracy_score
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time() # Get end time
    
    # Calculate the training time
    results['train_time'] = end - start
        
    # Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end - start
            
    # Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train[:300],predictions_train)
        
    # Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test,predictions_test)
    
    # Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train[:300],predictions_train,beta=0.5)
        
    # Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test,predictions_test,beta=0.5)
    
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    # Return the results
    return results