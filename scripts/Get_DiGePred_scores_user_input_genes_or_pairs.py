#!/usr/bin/env python

import pandas as pd  # version 0.25.1
import numpy as np
import _pickle as pickle
import networkx as nx  # version 1.9
import argparse
import itertools
import datetime

pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 5000)

now = datetime.datetime.now()
month = str(now.strftime("%m"))
day = str(now.strftime("%d"))
year = str(now.strftime("%Y"))
hour = str(now.strftime("%H"))
minute = str(now.strftime("%M"))

parser = argparse.ArgumentParser(description='Get DiGePred results')

parser.add_argument('-g', '--genes', type=str,
                    help='user input genes to get DiGePred scores',
                    dest='genes', required=False,
                    metavar='')

parser.add_argument('-p', '--pairs', type=str,
                    help='user input gene pairs to get DiGePred scores',
                    dest='pairs', required=False,
                    metavar='')

parser.add_argument('-m', '--model', type=str, default='unaffected no gene overlap',
                    help='DiGePred model to be used. If not provided, the best performing "unaffected-no-gene-overlap" model will be used',
                    dest='model', required=False,
                    metavar='')

parser.add_argument('-n', '--name', type=str,
                    default='{y}-{m}-{d}-{hr}{mi}'.format(m=month, d=day, y=year, hr=hour, mi=minute),
                    help='project name to be specified by user. If not provided, output file will be named with current date and time.',
                    dest='name', required=False,
                    metavar='')
parser.add_argument('-w_p', '--with-phenotype', action='store_true', help='The model is trained with all features.')
parser.add_argument('-wo_p', '--without-phenotype', dest='remove_phen_features', action='store_true', help='The model is trained without the phenotypes features')
parser.set_defaults(remove_phen_features=False)
parser.add_argument('-path', '--path-to-folder', dest='path_folder', help='Path where the folder DiGePred is stored, ex: /Users/Desktop', required=True, type=str)
parser.add_argument('-d', '--date-model-name', dest='date', help='date of creation of the re-trained model. Format MMMDD_YY ex: Jul25_222', required=True, type=str)

args = vars(parser.parse_args())


## Load pathway data files
reactome_gene_to_path_codes = pickle.load(open(args["path_folder"]+'/DiGePred/data/pathways/reactome/reactome_gene_to_path_codes.txt', 'rb'))
kegg_gene_to_path_codes = pickle.load(open(args["path_folder"]+'/DiGePred/data/pathways/kegg/kegg_gene_to_path_codes.txt', 'rb'))

## Load phenotype data files
hpo_gene_to_code = pickle.load(open(args["path_folder"]+'/DiGePred/data/phenotypes/hpo/hpo_gene_to_code.txt', 'rb'))

## Load co-expression data files
coexpress_dict = pickle.load(open(args["path_folder"]+'/DiGePred/data/coex/mutual_co-expression_rank_dict.txt', 'rb'))

## Load network data files
G_ppi = nx.read_dot(args["path_folder"]+'/DiGePred/data/networks/UCSC_ppi_network_new.dot')
G_pwy = nx.read_dot(args["path_folder"]+'/DiGePred/data/networks/UCSC_pwy_network_new.dot')
G_txt = nx.read_dot(args["path_folder"]+'/DiGePred/data/networks/UCSC_txt_network_new.dot')

dists_ppi = pickle.load(open(args["path_folder"]+'/DiGePred/data/networks/PPI_network_all_pairs_shortest_paths_Feb21_19.pkl', 'rb'))
dists_pwy = pickle.load(open(args["path_folder"]+'/DiGePred/data/networks/PWY_network_all_pairs_shortest_paths_Feb21_19.pkl', 'rb'))
dists_txt = pickle.load(open(args["path_folder"]+'/DiGePred/data/networks/Txt_network_all_pairs_shortest_paths_Feb21_19.pkl', 'rb'))

## Load evoltuonary biology and genomics feature data files
lof_dict = pickle.load(open(args["path_folder"]+'/DiGePred/data/evolgen/lof_pli_dict.pickle', 'rb'))
hap_insuf_dict = pickle.load(open(args["path_folder"]+'/DiGePred/data/evolgen/happloinsufficiency_dict.pickle', 'rb'))
protein_age_dict = pickle.load(open(args["path_folder"]+'/DiGePred/data/evolgen/protein_age_dict.pickle', 'rb'))
dNdS_avg_dict = pickle.load(open(args["path_folder"]+'/DiGePred/data/evolgen/dNdS_avg.pickle', 'rb'))
gene_ess_dict = pickle.load(open(args["path_folder"]+'/DiGePred/data/evolgen/Gene_Essentiality_dict.txt', 'rb'))


genes = []
pairs = []

if parser.parse_args().pairs:
    pairs_list = open(args["path_folder"] +"/"+ parser.parse_args().pairs).read().split('\n')[:-1]

    for line in pairs_list:
        g1 = line.split(',')[0].strip().rstrip()
        g2 = line.split(',')[1].strip().rstrip()

        pairs.append(tuple(sorted([g1, g2])))

    pairs = sorted(set(pairs))

elif parser.parse_args().genes:

    genes = sorted(set(open(parser.parse_args().genes).read().split('\n')[:-1]))
    pairs = itertools.combinations(set(genes), 2)

else:
    print('No input file found!')

project_name = parser.parse_args().name
model = parser.parse_args().model


## Load DiGePred models

clfs = dict()
if args['remove_phen_features']:
    model_name = args["path_folder"] + '/output/retrained_models/without_phenotype_features_'
else:
    model_name = args["path_folder"] + '/output/retrained_models/'

clfs['permuted'] = pd.read_pickle(model_name+'permuted_'+args['date'] + ".sav")
clfs['random'] = pd.read_pickle(model_name +'random_'+args['date'] + ".sav")
clfs['matched'] = pd.read_pickle(model_name +'matched_'+args['date'] + ".sav")
clfs['unaffected'] = pd.read_pickle(model_name +'unaffected_'+args['date'] + ".sav")
#clfs['all-digenic-vs-unaffected'] = pd.read_pickle(model_name +'all-digenic-vs-unaffected_'+args['date'] + ".sav")
clfs['unaffected-no-gene-overlap'] = pd.read_pickle(model_name+'unaffected no gene overlap_'+args['date'] + ".sav")
clfs['random-no-gene-overlap'] = pd.read_pickle(model_name +'random no gene overlap_'+args['date'] + ".sav")


## Function to get feature values as a pandas dataframe
def get_features(pairs):

    new_list_pairs = [tuple(sorted(p)) for p in list(set(pairs))]

    all_data = []

    for x in new_list_pairs:

        data = np.empty((1, 21))

        #  Pathway
        path1 = []
        path2 = []

        if x[0] in kegg_gene_to_path_codes or x[0] in reactome_gene_to_path_codes:

            if x[0] in kegg_gene_to_path_codes:
                path1 = kegg_gene_to_path_codes[x[0]]

            if x[0] in reactome_gene_to_path_codes:
                path1.extend(reactome_gene_to_path_codes[x[0]])

        if x[1] in kegg_gene_to_path_codes or x[1] in reactome_gene_to_path_codes:

            if x[1] in kegg_gene_to_path_codes:
                path2 = kegg_gene_to_path_codes[x[1]]

            if x[1] in reactome_gene_to_path_codes:
                path2.extend(reactome_gene_to_path_codes[x[1]])

        total = list(set(path1).union(path2))
        common = list(set(path1).intersection(path2))

        vqm = np.sqrt((len(path1) ** 2 + len(path2) ** 2) / 2)
        data[0][0] = vqm

        if len(total) == 0:
            data[0][1] = 0.
        else:
            data[0][1] = float(len(common)) / len(total)

        # HPO
        hpo1 = []
        hpo2 = []
        if x[0] in hpo_gene_to_code:
            hpo1 = hpo_gene_to_code[x[0]]
        if x[1] in hpo_gene_to_code:
            hpo2 = hpo_gene_to_code[x[1]]
        total = list(set(hpo1).union(hpo2))
        common = list(set(hpo1).intersection(hpo2))
        vqm = np.sqrt((len(hpo1) ** 2 + len(hpo2) ** 2) / 2)

        data[0][2] = vqm

        if len(total) == 0:
            data[0][3] = 0.
        else:
            data[0][3] = float(len(common)) / len(total)

        # PPI Network
        dist = []
        neighbors1 = []
        neighbors2 = []
        if x[0] in dists_ppi:
            neighbors1 = [p for p in nx.all_neighbors(G_ppi, x[0]) if p != x[0]]
            if x[1] in dists_ppi[x[0]]:
                dist.append(dists_ppi[x[0]][x[1]])
        if x[1] in dists_ppi:
            neighbors2 = [p for p in nx.all_neighbors(G_ppi, x[1]) if p != x[1]]
            if x[0] in dists_ppi[x[1]]:
                dist.append(dists_ppi[x[1]][x[0]])
        if dist != [] and min(dist) > 0:
            ppi_dist = 1 / float(min(dist))
        else:
            ppi_dist = 0.

        total = list(set(neighbors1).union(neighbors2))
        common = list(set(neighbors1).intersection(neighbors2))
        vqm = np.sqrt((len(neighbors1) ** 2 + len(neighbors2) ** 2) / 2)

        data[0][4] = vqm

        if len(total) == 0:
            data[0][5] = 0.
        else:
            data[0][5] = float(len(common)) / len(total)

        # data[i][8] = len(common)
        data[0][6] = ppi_dist

        # PWY Network
        dist = []
        neighbors1 = []
        neighbors2 = []
        if x[0] in dists_pwy:
            neighbors1 = [p for p in nx.all_neighbors(G_pwy, x[0]) if p is not x[0]]
            if x[1] in dists_pwy[x[0]]:
                dist.append(dists_pwy[x[0]][x[1]])
        if x[1] in dists_pwy:
            neighbors2 = [p for p in nx.all_neighbors(G_pwy, x[1]) if p is not x[1]]
            if x[0] in dists_pwy[x[1]]:
                dist.append(dists_pwy[x[1]][x[0]])

        if dist != [] and min(dist) > 0:
            pwy_dist = 1 / float(min(dist))
        else:
            pwy_dist = 0.

        total = list(set(neighbors1).union(neighbors2))
        common = list(set(neighbors1).intersection(neighbors2))
        vqm = np.sqrt((len(neighbors1) ** 2 + len(neighbors2) ** 2) / 2)

        data[0][7] = vqm

        if len(total) == 0:
            data[0][8] = 0.
        else:
            data[0][8] = float(len(common)) / len(total)

        # data[i][12] = len(common)
        data[0][9] = pwy_dist

        # TXT Network
        dist = []
        neighbors1 = []
        neighbors2 = []
        if x[0] in dists_txt:
            neighbors1 = [p for p in nx.all_neighbors(G_txt, x[0]) if p is not x[0]]
            if x[1] in dists_txt[x[0]]:
                dist.append(dists_txt[x[0]][x[1]])
        if x[1] in dists_txt:
            neighbors2 = [p for p in nx.all_neighbors(G_txt, x[1]) if p is not x[1]]
            if x[0] in dists_txt[x[1]]:
                dist.append(dists_txt[x[1]][x[0]])

        if dist != [] and min(dist) > 0:
            txt_dist = 1 / float(min(dist))
        else:
            txt_dist = 0.

        total = list(set(neighbors1).union(neighbors2))
        common = list(set(neighbors1).intersection(neighbors2))
        vqm = np.sqrt((len(neighbors1) ** 2 + len(neighbors2) ** 2) / 2)

        data[0][10] = vqm

        if len(total) == 0:
            data[0][11] = 0.
        else:
            data[0][11] = float(len(common)) / len(total)

        # data[i][16] = len(common)
        data[0][12] = txt_dist

        # Co-expression

        rankcoex1 = []
        rankcoex2 = []
        coexvalue = 0.
        if x[0] in coexpress_dict:
            rankcoex1 = [c for c in coexpress_dict[x[0]] if coexpress_dict[x[0]][c] < 100]
            if x[1] in coexpress_dict[x[0]]:
                coexvalue = 1 / coexpress_dict[x[0]][x[1]]
        if x[1] in coexpress_dict:
            rankcoex2 = [c for c in coexpress_dict[x[1]] if coexpress_dict[x[1]][c] < 100]
            if x[0] in coexpress_dict[x[1]]:
                coexvalue = 1 / coexpress_dict[x[1]][x[0]]

        total = list(set(rankcoex1).union(rankcoex2))
        common = list(set(rankcoex1).intersection(rankcoex2))
        vqm = np.sqrt((len(rankcoex1) ** 2 + len(rankcoex2) ** 2) / 2)

        data[0][13] = vqm

        if len(total) == 0:
            data[0][14] = 0.
        else:
            data[0][14] = float(len(common)) / len(total)

        # data[i][20] = len(common)
        data[0][15] = coexvalue

        # Lof

        if x[0] in lof_dict:
            v1 = lof_dict[x[0]]
        else:
            v1 = 0

        if x[1] in lof_dict:
            v2 = lof_dict[x[1]]
        else:
            v2 = 0

        vqm = np.sqrt((v1 ** 2 + v2 ** 2) / 2)
        data[0][16] = vqm

        # Happloinsufficiency Analysis

        if x[0] in hap_insuf_dict:
            v1 = hap_insuf_dict[x[0]]
        else:
            v1 = 0

        if x[1] in hap_insuf_dict:
            v2 = hap_insuf_dict[x[1]]
        else:
            v2 = 0

        vqm = np.sqrt((v1 ** 2 + v2 ** 2) / 2)
        data[0][17] = vqm

        # Protein Age

        if x[0] in protein_age_dict:
            v1 = protein_age_dict[x[0]]
        else:
            v1 = 0

        if x[1] in protein_age_dict:
            v2 = protein_age_dict[x[1]]
        else:
            v2 = 0

        vqm = np.sqrt((v1 ** 2 + v2 ** 2) / 2)
        data[0][18] = vqm

        # dN/DS

        if x[0] in dNdS_avg_dict:
            v1 = dNdS_avg_dict[x[0]]
        else:
            v1 = 0

        if x[1] in dNdS_avg_dict:
            v2 = dNdS_avg_dict[x[1]]
        else:
            v2 = 0

        vqm = np.sqrt((v1 ** 2 + v2 ** 2) / 2)
        data[0][19] = vqm

        # Gene Essentiality

        if x[0] in gene_ess_dict:
            v1 = np.mean(gene_ess_dict[x[0]])
        else:
            v1 = 0.
        if x[1] in gene_ess_dict:
            v2 = np.mean(gene_ess_dict[x[1]])
        else:
            v2 = 0.

        vqm = np.sqrt((v1 ** 2 + v2 ** 2) / 2)
        data[0][20] = vqm

    df = pd.DataFrame(all_data, index=new_list_pairs, columns=[
        # Pathways
        '#ofpathways',  # 0
        'common_pathways',  # 1
        # Phenotypes
        '#ofphenotypes',  # 2
        'common_phenotypes',  # 3
        # PPI network
        '#ofNeighborsPPI',  # 4
        '#Common_PPI_Neighbors',  # 5
        'PPI_network_dist',  # 6
        # PWY network
        '#ofNeighborsPWY',  # 7
        '#common_PWY_neighbors',  # 8
        'PWY_network_dist',  # 9
        # Txt network
        '#ofNeighborsTxt',  # 10
        '#Common_Txt_Neighbors',  # 11
        'Txt_network_dist',  # 12
        # Co-expression
        '#ofHighlyCoexpressed',  # 13
        '#Common_coexpressed',  # 14
        'Co-expression_coefficient',  # 15
        # LoF
        'LoFintolerance',  # 16
        # Haploinsuffiency
        'Haploinsufficiency',  # 17
        # Protein Age
        'protein_Age',  # 18
        # dN/dS
        'dN/dS',  # 19
        # Gene Essentiality
        'Essentiality',  # 20

    ])

    if args['remove_phen_features']:
        print("before df shape:", df.shape)
        df.drop("common_phenotypes", axis=1, inplace=True)
        df.drop("#ofPhenotypeCodes_combined", axis=1, inplace=True)
        print("phenotype features removed", df.shape)

    return df


if __name__ == '__main__':

    digepred_res_df = get_features(pairs)  # get feature values and save in a pandas DiGePred results df.
    p = clfs[model].predict_proba(digepred_res_df)[:, 1]  # get predictions based on DiGePred model specified.
    digepred_res_df[model] = p  # add column to DiGePred result df.

    digepred_res_df.to_csv(args["path_folder"]+'/output/scores_pairs_input/{resultat_{project_name}.csv'.format(project_name=project_name),
                    sep=',', header=True, index=False)  # save feature values and predictions as DiGePred results CSV.

