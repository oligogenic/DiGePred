from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np  # numpy == 1.19.5
import pandas as pd  # version 0.25.1
import datetime
import random
import pickle
import argparse

now = datetime.datetime.now()
month = str(now.strftime("%b"))
day = str(now.strftime("%d"))
year = str(now.strftime("%y"))

parser = argparse.ArgumentParser()
parser.add_argument('-w_p', '--with-phenotype', action='store_true', help='The model is trained with all features.')
parser.add_argument('-wo_p', '--without-phenotype', dest='remove_phen_features', action='store_true',
                    help='The model is trained without the phenotypes features')
parser.set_defaults(remove_phen_features=False)
parser.add_argument('-p', '--path-to-folder', dest='path_folder',
                    help='Path where the folder DiGePred is stored, ex: /Users/Desktop', required=True, type=str)
args = vars(parser.parse_args())

sel_feats = ['common_pathways',
             'common_phenotypes',
             'Co-expression_coefficient',
             'PPI_network_dist',
             'PWY_network_dist',
             'Txt_network_dist',
             'LoFintolerance_combined',
             'Haploinsufficiency_combined',
             'Age_combined',
             'dN/dS_combined',
             'Essentiality_combined',
             '#ofpathways_combined',
             '#ofPhenotypeCodes_combined',
             '#ofNeighborsPPI_combined',
             '#ofNeighborsPWY_combined',
             '#ofNeighborsTxt_combined',
             '#ofHighlyCoexpressed_combined',
             '#Common_PPI_Neighbors',
             '#common_PWY_neighbors',
             '#Common_Txt_Neighbors',
             '#Common_coexpressed',
             ]

# Import dfs
digenic_training = pd.read_csv(args["path_folder"] + "/DiGePred/positives/training/digenic_DIDA_pairs_training.csv",
                               index_col=0)
digenic_training_no_overlap = pd.read_csv(
    args["path_folder"] + '/DiGePred/positives/training/digenic_DIDA_pairs_no-gene-overlap_training.csv', index_col=0)
unaffected_non_digenic_training = pd.read_csv(
    args["path_folder"] + '/DiGePred/negatives/training/unaffected_non_digenic_pairs_training.csv', index_col=0)
random_non_digenic_training = pd.read_csv(
    args["path_folder"] + '/DiGePred/negatives/training/random_non_digenic_pairs_training.csv', index_col=0)
permuted_non_digenic_training = pd.read_csv(
    args["path_folder"] + '/DiGePred/negatives/training/permuted_non_digenic_pairs_training.csv', index_col=0)
matched_non_digenic_training = pd.read_csv(
    args["path_folder"] + '/DiGePred/negatives/training/matched_non_digenic_pairs_training.csv', index_col=0)
unaffected_no_gene_overlap_non_digenic_training = pd.read_csv(
    args["path_folder"] + '/DiGePred/negatives/training/unaffected-no-gene-overlap_non_digenic_pairs_training.csv',
    index_col=0)
random_no_gene_overlap_non_digenic_training = pd.read_csv(
    args["path_folder"] + '/DiGePred/negatives/training/random-no-gene-overlap_non_digenic_pairs_training.csv',
    index_col=0)

list_dfs = [digenic_training, digenic_training_no_overlap, unaffected_non_digenic_training,
            random_non_digenic_training, permuted_non_digenic_training, matched_non_digenic_training,
            unaffected_no_gene_overlap_non_digenic_training, random_no_gene_overlap_non_digenic_training]
for dataframe in list_dfs:
    if args['remove_phen_features']:
        dataframe.drop("common_phenotypes", axis=1, inplace=True)
        dataframe.drop("#ofPhenotypeCodes_combined", axis=1, inplace=True)

# define models with correpsonding positive and negative sets

models = {'unaffected': {'pos': digenic_training,
                         'neg': unaffected_non_digenic_training},
          'permuted': {'pos': digenic_training,
                       'neg': permuted_non_digenic_training},
          'random': {'pos': digenic_training,
                     'neg': random_non_digenic_training},
          'matched': {'pos': digenic_training,
                      'neg': matched_non_digenic_training},
          'unaffected no gene overlap': {'pos': digenic_training_no_overlap,
                                         'neg': unaffected_no_gene_overlap_non_digenic_training},
          'random no gene overlap': {'pos': digenic_training_no_overlap,
                                     'neg': random_no_gene_overlap_non_digenic_training},
          }

roc_aucs = {}
pr_aucs = {}
f1_scores = {}
tprs = {}
fprs = {}
precisions = {}
recalls = {}
test_data = {}
pred_probs = {}

pos_neg_ratio = 75

# Looping through all models


for m in models:

    digenic_pairs_df = models[m]['pos']
    digenic_pairs_list = list(digenic_pairs_df.index)

    non_digenic_pairs_df = models[m]['neg']
    non_digenic_pairs_list = list(non_digenic_pairs_df.index)

    roc_aucs[m] = []
    pr_aucs[m] = []
    f1_scores[m] = []

    tprs[m] = []
    fprs[m] = []
    precisions[m] = []
    recalls[m] = []

    test_data[m] = []
    pred_probs[m] = []

    # Making full set by adding digenic and non-digenic sets

    full_set_X = pd.concat([digenic_pairs_df, non_digenic_pairs_df], ignore_index=True)
    full_set_y = np.asarray([1] * len(digenic_pairs_list) + [0] * len(non_digenic_pairs_list))

    sss_splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_jobs=1, n_estimators=500, max_depth=15)

    for train, test in sss_splitter.split(full_set_X, full_set_y):
        X_train, X_test, y_train, y_test = full_set_X.iloc[train], \
                                           full_set_X.iloc[test], \
                                           full_set_y[train], \
                                           full_set_y[test]

        # Train classifier using inner train

        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)
        predictions = clf.predict(X_test)

        # Evaluate performance using ROC, PR AUCs and F1 scores

        fpr, tpr, thresholds = roc_curve(y_test, preds[:, 1])
        roc_auc = auc(fpr, tpr)

        p, r, _ = precision_recall_curve(y_test, preds[:, 1])
        ap = average_precision_score(y_test, preds[:, 1])
        # f1 = (2 * p * r) / (p + r)

        f1 = f1_score(y_test, predictions)

        roc_aucs[m].append(roc_auc)
        pr_aucs[m].append(ap)
        f1_scores[m].append(f1)

        tprs[m].append(tpr)
        fprs[m].append(fpr)
        precisions[m].append(p)
        recalls[m].append(r)

        test_data[m].append(y_test)
        pred_probs[m].append(preds[:, 1])

    # Saving model
    clf = RandomForestClassifier(n_jobs=1, n_estimators=500, max_depth=15)
    # Fit the model on training set
    clf.fit(full_set_X, full_set_y)
    # save the model to disk
    if args['remove_phen_features']:
        pickle.dump(clf, open(args[
                                  "path_folder"] + '/output/retrained_models/' + "without_phenotype_features_" + m + '_{month}{day}_{year}.sav'.format(
            month=month, day=day, year=year), 'wb'))
    else:
        pickle.dump(clf, open(
            args["path_folder"] + '/output/retrained_models/' + m + '_{month}{day}_{year}.sav'.format(month=month,
                                                                                                      day=day,
                                                                                                      year=year), 'wb'))

data_cols = {
    'ROC_AUCs': roc_aucs,
    'PR_AUCs': pr_aucs,
    'F1_scores': f1_scores,
    'TPRs': tprs,
    'FPRs': fprs,
    'Precisions': precisions,
    'Recalls': recalls,
    'Test data': test_data,
    'Pred probs': pred_probs
}

df = pd.DataFrame(index=list(models), columns=data_cols.keys())

for m in models:
    for d in data_cols:
        df[d][m] = data_cols[d][m]

if args['remove_phen_features']:
    df.to_pickle(args[
                     "path_folder"] + '/output/training_performance/without_phenotype_features_DiGePred_training_performance_{month}{day}_{year}.pkl'.format(
        month=month
        , day=day,
        year=year))
else:
    df.to_pickle(args[
                     "path_folder"] + '/output/training_performance/DiGePred_training_performance_{month}{day}_{year}.pkl'.format(
        month=month, day=day, year=year))
