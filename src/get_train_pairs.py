
import numpy as np
import pickle as pkl
import argparse
from utils.sample import sample_pairs
from utils.functions import dist_labels_to_changepoint_labels
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import random
import torch
from utils.create_features import CreateFeatures

def create_graphs_with_features(all_graphs, args):
    """
    Creates train/val/test pairs of graphs. Each pair contains two networks, each from different years, and a label of 0/1 which represents how similar they are. 
    0: There is a change-point between them (because they belong to different distributions)
    1: There is not a change-point between them (because they belong to the same distribution)

    The data is balanced because there tend to be more negative pairs than positive.
    """

    crisis_years = [1973, 1991, 1995, 2000, 2007, 2016, 2018]
    phases = []
    p = -1
    for i in range(1962,2019):
        if i in crisis_years:
            p += 1
        phases.append(p)

    labels = dist_labels_to_changepoint_labels(phases)
    graph_pairs = sample_pairs(all_graphs,labels)

    with open(f"{args.save_dir}/graph_pairs_{args.gdp}_{args.feature_type}", 'wb') as f:
        pkl.dump(graph_pairs, f)

    train_indices, test_indices = train_test_split(np.arange(len(graph_pairs)), test_size=0.40, random_state=42)
    test_indices, val_indices = train_test_split(test_indices, test_size=0.5, random_state=42)

    graph_pairs_train = [graph_pairs[i] for i in train_indices]
    graph_pairs_test = [graph_pairs[i] for i in test_indices]
    graph_pairs_val = [graph_pairs[i] for i in val_indices]


    positive_samples = [item for item in graph_pairs_train if item[2] == 1]
    negative_samples = [item for item in graph_pairs_train if item[2] == 0]

    # Calculate the difference in count
    diff = len(negative_samples) - len(positive_samples)

    # Upsample positive samples
    if diff > 0:
        positive_samples_upsampled = resample(positive_samples, replace=True, n_samples=len(negative_samples), random_state=42)
        balanced_data = negative_samples + positive_samples_upsampled
    # Upsample negative samples
    elif diff < 0:
        negative_samples_upsampled = resample(negative_samples, replace=True, n_samples=len(positive_samples), random_state=42)
        balanced_data = negative_samples_upsampled + positive_samples
    else:
        balanced_data = graph_pairs_train

    random.shuffle(balanced_data)

    with open(f'{args.save_dir}/train_data_{args.gdp}_{args.feature_type}.pkl', 'wb') as f:
        pkl.dump(balanced_data, f)

    with open(f'{args.save_dir}/val_data_{args.gdp}_{args.feature_type}.pkl', 'wb') as f:
        pkl.dump(graph_pairs_val, f)
        
    with open(f'{args.save_dir}/test_data_{args.gdp}_{args.feature_type}.pkl', 'wb') as f:
        pkl.dump(graph_pairs_test, f)


def add_features(args):
    """
    World bank data are added to the networks for each year as node features.
    """

    with open(f"{args.graph_dir}", "rb") as f:         
        graphs = pkl.load(f)

    if args.feature_type == 'none':
        create_graphs_with_features(graphs, args)
    else:

        all_nodes = ['ABW', 'AFG', 'AGO', 'ALB', 'AND', 'ARE', 'ARG', 'ARM', 'ASM',
        'ATG', 'AUS', 'AUT', 'AZE', 'BDI', 'BEL', 'BEN', 'BFA', 'BGD',
        'BGR', 'BHR', 'BHS', 'BIH', 'BLR', 'BLZ', 'BMU', 'BOL', 'BRA',
        'BRB', 'BRN', 'BTN', 'BWA', 'CAF', 'CAN', 'CHE', 'CHL', 'CHN',
        'CIV', 'CMR', 'COD', 'COG', 'COL', 'COM', 'CPV', 'CRI', 'CUB',
        'CUW', 'CYM', 'CYP', 'CZE', 'DEU', 'DMA', 'DNK', 'DOM', 'DZA',
        'ECU', 'EGY', 'ESP', 'EST', 'ETH', 'FIN', 'FJI', 'FRA', 'FSM',
        'GAB', 'GBR', 'GEO', 'GHA', 'GIN', 'GMB', 'GNB', 'GNQ', 'GRC',
        'GRD', 'GRL', 'GTM', 'GUM', 'GUY', 'HKG', 'HND', 'HRV', 'HTI',
        'HUN', 'IDN', 'IND', 'IRL', 'IRN', 'IRQ', 'ISL', 'ISR', 'ITA',
        'JAM', 'JOR', 'JPN', 'KAZ', 'KEN', 'KGZ', 'KHM', 'KNA', 'KOR',
        'KWT', 'LAO', 'LBN', 'LBR', 'LBY', 'LCA', 'LKA', 'LSO', 'LTU',
        'LUX', 'LVA', 'MAC', 'MAR', 'MDA', 'MDG', 'MDV', 'MEX', 'MHL',
        'MKD', 'MLI', 'MLT', 'MMR', 'MNE', 'MNG', 'MNP', 'MOZ', 'MRT',
        'MUS', 'MWI', 'MYS', 'NAM', 'NER', 'NGA', 'NIC', 'NLD', 'NOR',
        'NPL', 'NRU', 'NZL', 'OMN', 'PAK', 'PAN', 'PER', 'PHL', 'PLW',
        'PNG', 'POL', 'PRT', 'PRY', 'PSE', 'PYF', 'QAT', 'ROU', 'RUS',
        'RWA', 'SAU', 'SDN', 'SEN', 'SGP', 'SLB', 'SLE', 'SLV', 'SMR',
        'SRB', 'SSD', 'STP', 'SUR', 'SVK', 'SVN', 'SWE', 'SWZ', 'SXM',
        'SYC', 'SYR', 'TCD', 'TGO', 'THA', 'TJK', 'TKM', 'TLS', 'TON',
        'TTO', 'TUN', 'TUR', 'TUV', 'TZA', 'UGA', 'UKR', 'URY', 'USA',
        'UZB', 'VCT', 'VEN', 'VNM', 'VUT', 'WSM', 'YEM', 'ZAF', 'ZMB',
        'ZWE']
            
        with open(f"feature_dicts/{args.feature_type}.pkl", "rb") as f:
            feat_dict = pkl.load(f)

        years = range(1962,2019)

        if args.gdp == 'nogdp':
            dim = 25
        else:
            dim = 27
            
        zeros = torch.zeros(dim)

        for i in range(len(years)):
            new_x = torch.empty(0, dim)
            year = years[i]

            feat_dict_year = feat_dict[year].combined_features

            for j, country in enumerate(all_nodes):
                if j == 0:
                    new_x = torch.stack([zeros])

                elif country in feat_dict_year["country_code"].values:
                    tensor_before = graphs[i].x[j]
                    country_row = feat_dict_year[feat_dict_year["country_code"] == country]
                    country_row = country_row.drop(columns = ["country_code", "current_gdp_growth"])
                    row_values = country_row.values.tolist()
                    row_tensor = torch.tensor(row_values)[0]
                    combined_values = torch.cat((tensor_before, row_tensor))

                    new_x = torch.cat((new_x, combined_values.unsqueeze(0)), dim=0)

                else:
                    new_x = torch.cat((new_x, zeros.unsqueeze(0)), dim=0)

            graphs[i].x = new_x
            
        create_graphs_with_features(graphs, args)

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir', type=str, default='data/graphs/graphs_gdp.pkl', help='Name of folder with initial graphs')
    parser.add_argument('--save_dir', type=str, default='data/train-data/gdp_mis_norm', help='Name of folder where to store results')
    parser.add_argument('--gdp', type=str, default='gdp', help='Whether GDP is included')
    parser.add_argument('--feature_type', type=str, default='mis_norm', help='Types of features to be added')

    args = parser.parse_args()
    
  
    return args

def main():
    args = get_args()

    add_features(args)

if __name__ == '__main__':
    main()