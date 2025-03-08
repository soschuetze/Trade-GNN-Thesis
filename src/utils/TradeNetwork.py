from sklearn.preprocessing import StandardScaler
import pickle as pkl
import pandas as pd
from tqdm import trange
import torch
import torch_geometric.datasets as datasets
import torch_geometric.data as data
import torch_geometric.transforms as transforms
import requests
import numpy as np


def get_sitc_codes():
    # URL of the JSON file
    url = 'https://comtradeapi.un.org/files/v1/app/reference/S4.json'

    try:
        # Send a GET request to the URL and fetch the data
        response = requests.get(url)
        response.raise_for_status()  # Check that the request was successful
        
        # Load the JSON data
        data = response.json()

        # Since the JSON data might be nested, use json_normalize with appropriate arguments
        if isinstance(data, list):
            # If the top level is a list
            df = pd.json_normalize(data)
        else:
            # If the top level is a dictionary
            # Identify the key that holds the main data (adjust the path as necessary)
            main_data_key = 'results'  # Adjust this based on the actual structure
            df = pd.json_normalize(data[main_data_key])

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    except ValueError as e:
        print(f"Error parsing JSON: {e}")
    except KeyError as e:
        print(f"Error processing JSON structure: {e}")

    return df

class TradeNetwork:
    """
    We define a class which computes the MST trade network for a given year 
    """
    
    def __init__(self, year = 1962, data_dir = "data"):
        self.year = year
        self.data_dir = data_dir
        
    def prepare_features(self, filter_gdp = True):
        
        ###IMPORT GDP###
        #prepare GDP as a set of features 
        with open('data/all_wb_indicators.pickle', 'rb') as handle:
            features_dict = pkl.load(handle)

        self.gdp = features_dict['NY.GDP.MKTP.CD']
        scaler = StandardScaler()
        self.gdp[["prev_gdp"]] = scaler.fit_transform(np.log(self.gdp[['YR'+str(self.year-1)]]))
        self.gdp[["current_gdp"]] = scaler.fit_transform(np.log(self.gdp[['YR'+str(self.year)]]))
        #rename and keep relevant columns
        self.gdp["country_code"] = self.gdp["economy"]
        self.gdp = self.gdp[["country_code", "prev_gdp", "current_gdp"]].dropna()
        
        ###IMPORT GDP GROWTH###
        #prepare GDP growth
        self.gdp_growth = features_dict['NY.GDP.MKTP.KD.ZG']
        self.gdp_growth["prev_gdp_growth"] = self.gdp_growth['YR'+str(self.year-1)]
        self.gdp_growth["current_gdp_growth"] = self.gdp_growth['YR'+str(self.year)] 
        self.gdp_growth["future_gdp_growth"] = self.gdp_growth['YR'+str(self.year+1)]
        #rename and keep relevant columns
        self.gdp_growth["country_code"] = self.gdp_growth["economy"]
        self.gdp_growth = self.gdp_growth[["country_code", "prev_gdp_growth",
                                "current_gdp_growth", "future_gdp_growth"]].dropna()
        
        ###IMPORT GDP PER CAPITA###
        self.gdp_per_capita = features_dict['NY.GDP.PCAP.CD']
        self.gdp_per_capita["prev_gdp_per_cap"] = self.gdp_per_capita['YR'+str(self.year-1)]
        self.gdp_per_capita["current_gdp_per_cap"] = self.gdp_per_capita['YR'+str(self.year)]
        self.gdp_per_capita["future_gdp_per_cap"] = self.gdp_per_capita['YR'+str(self.year+1)]
        #rename and keep relevant columns
        self.gdp_per_capita["country_code"] = self.gdp_per_capita["economy"]
        self.gdp_per_capita = self.gdp_per_capita[["country_code", "prev_gdp_per_cap",
                                "current_gdp_per_cap", "future_gdp_per_cap"]].dropna()
        
        ###IMPORT GDP PER CAPITA GROWTH###
        self.gdp_per_capita_growth = features_dict['NY.GDP.PCAP.KD.ZG']
        self.gdp_per_capita_growth["prev_gdp_per_cap_growth"] = self.gdp_per_capita_growth['YR'+str(self.year-1)]
        self.gdp_per_capita_growth["current_gdp_per_cap_growth"] = self.gdp_per_capita_growth['YR'+str(self.year)]
        self.gdp_per_capita_growth["future_gdp_per_cap_growth"] = self.gdp_per_capita_growth['YR'+str(self.year+1)]
        
        #rename and keep relevant columns
        self.gdp_per_capita_growth["country_code"] = self.gdp_per_capita_growth["economy"]
        self.gdp_per_capita_growth = self.gdp_per_capita_growth[["country_code", "prev_gdp_per_cap_growth",
                                "current_gdp_per_cap_growth", "future_gdp_per_cap_growth"]].dropna()
        
        ###MERGE ALL DATA FEATURES###
        self.features = pd.merge(self.gdp_growth, self.gdp, on = "country_code").dropna()
        self.features = pd.merge(self.features, self.gdp_per_capita, on = "country_code").dropna()
        self.features = pd.merge(self.features, self.gdp_per_capita_growth, on = "country_code").dropna()

    def prepare_network(self):
        """
        We create an initial, import-centric trade link pandas dataframe for a given year
        """
        #get product codes
        data_dict = get_sitc_codes()
        data_cross = []
        i = 0
        for item_def in list(data_dict["text"]):
            if(i >= 2):
                data_cross.append(item_def.split(" - ", 1))
            i = i+1

        self.product_codes = pd.DataFrame(data_cross, columns = ['code', 'product'])
        self.product_codes["sitc_product_code"] = self.product_codes["code"]
        
        #get country codes
        self.country_codes = pd.read_excel("data/ISO3166.xlsx")
        self.country_codes["location_code"] = self.country_codes["Alpha-3 code"]
        self.country_codes["partner_code"] = self.country_codes["Alpha-3 code"]
        self.country_codes["country_i"] = self.country_codes["English short name"]
        self.country_codes["country_j"] = self.country_codes["English short name"]
        
        #get trade data for a given year
        trade_data = pd.read_stata(self.data_dir + "/country_partner_sitcproduct4digit_year_"+ str(self.year)+".dta") 
        #merge with product / country descriptions
        trade_data = pd.merge(trade_data, self.country_codes[["location_code", "country_i"]],on = ["location_code"])
        trade_data = pd.merge(trade_data, self.country_codes[["partner_code", "country_j"]],on = ["partner_code"])
        trade_data = pd.merge(trade_data, self.product_codes[["sitc_product_code", "product"]], 
                              on = ["sitc_product_code"])
        ###select level of product aggregation
        trade_data["product_category"] = trade_data["sitc_product_code"].apply(lambda x: x[0:1])
        
        #keep only nodes that we have features for
        #trade_data = trade_data[trade_data["location_code"].isin(self.features["country_code"])]
        #trade_data = trade_data[trade_data["partner_code"].isin(self.features["country_code"])]
        
        if (len(trade_data.groupby(["location_code", "partner_code", "sitc_product_code"])["import_value"].sum().reset_index()) != len(trade_data)):
            print("import, export, product combination not unique!")
        self.trade_data1 = trade_data
        #from import-export table, create only import table
        #extract imports
        imports1 = trade_data[['location_id', 'partner_id', 'product_id', 'year',
               'import_value', 'sitc_eci', 'sitc_coi', 'location_code', 'partner_code',
               'sitc_product_code', 'country_i', 'country_j', 'product', "product_category"]]
        imports1 = imports1[imports1["import_value"] != 0]
        #transform records of exports into imports
        imports2 = trade_data[['location_id', 'partner_id', 'product_id', 'year',
               'export_value', 'sitc_eci', 'sitc_coi', 'location_code', 'partner_code',
               'sitc_product_code', 'country_i', 'country_j', 'product', "product_category"]]
        imports2["temp1"] = imports2['partner_code']
        imports2["temp2"] = imports2['location_code']

        imports2['location_code'] = imports2["temp1"]
        imports2['partner_code'] = imports2["temp2"]
        imports2["import_value"] = imports2["export_value"]
        imports2 = imports2[imports2["import_value"] != 0]
        imports2 = imports2[['location_id', 'partner_id', 'product_id', 'year',
               'import_value', 'sitc_eci', 'sitc_coi', 'location_code', 'partner_code',
               'sitc_product_code', 'country_i', 'country_j', 'product', "product_category"]]
        
        imports_table = pd.concat([imports1, imports2]).drop_duplicates()
        
        #rename columns for better clarity
        imports_table["importer_code"] = imports_table["location_code"]
        imports_table["exporter_code"] = imports_table["partner_code"]
        imports_table["importer_name"] = imports_table["country_i"]
        imports_table["exporter_name"] = imports_table["country_j"]
        
        cols = ["importer_code", "exporter_code", "importer_name", "exporter_name",
               'product_id', 'year', 'import_value', 'sitc_eci', 'sitc_coi',
               'sitc_product_code', 'product', "product_category"]
        imports_table = imports_table[cols]
        
        exporter_total = imports_table.groupby(["exporter_code"])["import_value"].sum().reset_index()
        exporter_total = exporter_total.rename(columns = {"import_value": "export_total"})
        
        importer_total = imports_table.groupby(["importer_code"])["import_value"].sum().reset_index()
        importer_total = importer_total.rename(columns = {"import_value": "import_total"})
        
        #sum imports across all products between countries into single value 
        imports_table_grouped = imports_table.groupby(["importer_code", "exporter_code"])["import_value"].sum().reset_index()
        
        #sum exports in each category 
        self.export_types = imports_table.groupby(["exporter_code", "product_category"])["import_value"].sum().reset_index()
        self.export_types = pd.merge(self.export_types, exporter_total, on = "exporter_code")
        #multiply by 100 to allow weights to scale better in GNN
        self.export_types["category_fraction"] = self.export_types.import_value/self.export_types.export_total*10
        ss = StandardScaler()
        columns = list(set(self.export_types["product_category"]))
        self.export_types = self.export_types[["exporter_code", "product_category", "category_fraction"]]\
        .pivot(index = ["exporter_code"], columns = ["product_category"], values = "category_fraction")\
        .reset_index().fillna(0)
        #rename columns
        rename_columns = []
        for col in self.export_types.columns:
            if(col == "exporter_code"):
                rename_columns.append(col)
            else:
                rename_columns.append("resource_" + col)
        self.export_types.columns = rename_columns
        self.export_types = self.export_types.rename(columns = {"exporter_code": "country_code"})
        self.features = pd.merge(self.features, self.export_types, 
                                on = "country_code", how = "left")
        
        #look at fraction of goods traded with each counterparty
        imports_table_grouped = pd.merge(imports_table_grouped, exporter_total, how = "left")
        imports_table_grouped["export_percent"] = imports_table_grouped["import_value"]/imports_table_grouped["export_total"]
        scaler = StandardScaler()
        imports_table_grouped[["export_percent_feature"]] = scaler.fit_transform(np.log(imports_table_grouped[["export_percent"]]))
        imports_table_grouped["export_percent_feature"] = imports_table_grouped["export_percent_feature"] + abs(min(imports_table_grouped["export_percent_feature"]))
        
        imports_table_grouped = pd.merge(imports_table_grouped, importer_total, how = "left")
        imports_table_grouped["import_percent"] = imports_table_grouped["import_value"]/imports_table_grouped["import_total"]
        scaler = StandardScaler()
        imports_table_grouped[["import_percent_feature"]] = scaler.fit_transform(np.log(imports_table_grouped[["import_percent"]]))
        imports_table_grouped["import_percent_feature"] = imports_table_grouped["import_percent_feature"] + abs(min(imports_table_grouped["import_percent_feature"]))
        
        self.trade_data = imports_table_grouped

    def graph_create(self, exporter = True,
            node_features = ['prev_gdp_growth', 'current_gdp_growth','prev_gdp','current_gdp'],
            node_labels = 'future_gdp_growth'):
        
        if(exporter):
            center_node = "exporter_code"
            neighbors = "importer_code"
            edge_features = 'export_percent'
        
        #filter features and nodes to ones that are connected to others in trade data
        # list_active_countries = list(set(list(self.trade_data ["importer_code"])+\
        #                 list(self.trade_data ["exporter_code"])))

        list_active_countries = ['ABW', 'AFG', 'AGO', 'ALB', 'AND', 'ARE', 'ARG', 'ARM', 'ASM',
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

        # Create a new DataFrame with the list of country codes
        df_new = pd.DataFrame(list_active_countries, columns=['country_code'])
        self.features = pd.merge(df_new, self.features, on='country_code', how='left')
        self.features = self.features.fillna(0)

        # self.features = self.features[self.features["country_code"].isin(list_active_countries)].reset_index()
        # self.features.fillna(0, inplace = True)
        self.features["node_numbers"] = self.features.index

        #create lookup dictionary making node number / node features combatible with ordering of nodes
        #in our edge table

        self.node_lookup1 = self.features.set_index('node_numbers').to_dict()['country_code']
        self.node_lookup2 = self.features.set_index('country_code').to_dict()['node_numbers']
        
        #get individual country's features
        self.regression_table = pd.merge(self.features, self.trade_data,
                        left_on = "country_code",
                        right_on = center_node, how = 'right')
        #get features for trade partners
        self.regression_table = pd.merge(self.features, self.regression_table,
                                        left_on = "country_code",
                                        right_on = neighbors, how = "right",
                                        suffixes = ("_neighbors", ""))
        
        self.trade_data = self.trade_data[self.trade_data[neighbors].isin(self.node_lookup2)]
        self.trade_data = self.trade_data[self.trade_data[center_node].isin(self.node_lookup2)]

        self.regression_table["source"] = self.trade_data[neighbors].apply(lambda x: self.node_lookup2[x])
        self.regression_table["target"] = self.trade_data[center_node].apply(lambda x: self.node_lookup2[x])    

        self.regression_table = self.regression_table.dropna()
        #filter only to relevant columns
        self.relevant_columns = ["source", "target"]
        self.relevant_columns.extend(node_features)
        self.relevant_columns.append(node_labels)
        self.graph_table = self.regression_table[self.relevant_columns]
        
        if(self.graph_table.isnull().values.any()): print("edges contain null / inf values")

        self.node_attributes = torch.tensor(np.array(self.features[node_features]))\
        .to(torch.float)
        self.source_nodes = list(self.graph_table["source"])
        self.target_nodes = list(self.graph_table["target"])

        self.edge_attributes = list(self.trade_data[edge_features])
        
        self.pyg_graph = data.Data(x = self.node_attributes,
                                   edge_index = torch.tensor([self.source_nodes, self.target_nodes]),
                                   edge_attr = torch.tensor(self.edge_attributes).to(torch.float),
                                   y = torch.tensor(list(self.features[node_labels])).to(torch.float))
        
        return self.pyg_graph