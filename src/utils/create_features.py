import pandas as pd
import pickle as pkl
import networkx as nx
import requests

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


class CreateFeatures:
    """
    We define a class which builds the feature dataframe 
    """
    
    def __init__(self, year = 1962, data_dir = "../data/"):
        self.year = year
        self.data_dir = data_dir
        
    def prepare_econ_features(self, filter_gdp = True):
        
        #DATA IMPORT
        #import dictionary with all features from WB
        with open(self.data_dir + 'all_wb_indicators.pickle', 'rb') as handle:
            features_dict = pkl.load(handle)
            
        self.feature_list = list(features_dict.keys())[1:]
        #import list of all features we want to select for

        #look up each of the features -- add country feature in that year 
        i = 0
        for feature in self.feature_list:
            #find dataframe corresponding to specific feature name
            df = features_dict[feature]
            
            if (i == 0):
                self.features = df[["economy", "YR" + str(self.year)]]
            else: 
                self.features = pd.merge(self.features, 
                                            df[["economy", "YR" + str(self.year)]],
                                            on = "economy", how = "outer")
            self.features.rename(columns = {"YR" + str(self.year): feature}, inplace = True)
            i = i+1
        
        #prepare GDP feature
        self.gdp_growth = features_dict['NY.GDP.MKTP.KD.ZG']
        cols = list(self.gdp_growth.columns.copy())
        cols.remove("economy")
        self.gdp_growth["country_sd"] = self.gdp_growth[cols].std(axis=1)
        #select potential variables 
        self.gdp_growth["prev_gdp_growth"] = self.gdp_growth["YR" + str(self.year-1)]
        self.gdp_growth["current_gdp_growth"] = self.gdp_growth["YR" + str(self.year)]
        #we eliminate countries that are too volatile in growth -- probably an indicator that growth estimates are inaccurate
        self.gdp_growth = self.gdp_growth[["economy", "prev_gdp_growth",
                                "current_gdp_growth"]].dropna()
        
        #combine GDP and other features
        self.features = pd.merge(self.gdp_growth, self.features,
                                   on = "economy", how = "left")
        #we only keep countries where we observe GDP growth -- otherwise nothing to predict
        #we keep countries where other features may be missing -- and fill NAs with 0 
        self.features.rename(columns = {"economy": "country_code"}, inplace = True)
        
    def prepare_network_features(self):
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
        self.country_codes = pd.read_excel(self.data_dir + "ISO3166.xlsx")
        self.country_codes["location_code"] = self.country_codes["Alpha-3 code"]
        self.country_codes["partner_code"] = self.country_codes["Alpha-3 code"]
        self.country_codes["country_i"] = self.country_codes["English short name"]
        self.country_codes["country_j"] = self.country_codes["English short name"]
        
        #get trade data for a given year
        trade_data = pd.read_stata(self.data_dir + "country_partner_sitcproduct4digit_year_"+ str(self.year)+".dta") 
        #merge with product / country descriptions
        trade_data = pd.merge(trade_data, self.country_codes[["location_code", "country_i"]],on = ["location_code"])
        trade_data = pd.merge(trade_data, self.country_codes[["partner_code", "country_j"]],on = ["partner_code"])
        trade_data = pd.merge(trade_data, self.product_codes[["sitc_product_code", "product"]], 
                              on = ["sitc_product_code"])
        ###select level of product aggregation
        trade_data["product_category"] = trade_data["sitc_product_code"].apply(lambda x: x[0:1])
        #trade_data = trade_data[trade_data["product_category"] == "1"]
        
        #keep only nodes that we have features for
        trade_data = trade_data[trade_data["location_code"].isin(self.features["country_code"])]
        trade_data = trade_data[trade_data["partner_code"].isin(self.features["country_code"])]
        
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
        
        ##### COMPUTE CENTRALITY FOR COUNTRY
        #sum imports across all products between countries into single value 
        imports_table_grouped = imports_table.groupby(["importer_code", "exporter_code"])["import_value"].sum().reset_index()
        imports_table_grouped = pd.merge(imports_table_grouped, importer_total, on = "importer_code")
        imports_table_grouped["import_fraction"] = imports_table_grouped["import_value"]\
                        /imports_table_grouped["import_total"]*100
        
        self.trade_data = imports_table_grouped
        
        #filter features and nodes to ones that are connected to others in trade data
        list_active_countries = list(set(list(self.trade_data ["importer_code"])+\
                        list(self.trade_data ["exporter_code"])))
        self.features = self.features[self.features["country_code"].isin(list_active_countries)].reset_index()
        self.features["node_numbers"] = self.features.index
        
        G=nx.from_pandas_edgelist(self.trade_data, 
                          "exporter_code", "importer_code", create_using = nx.DiGraph())
        
        self.G = G
        self.centrality_overall= nx.eigenvector_centrality(G, max_iter= 10000) 
        self.centrality_overall = pd.DataFrame(list(map(list, self.centrality_overall.items())), 
                                               columns = ["country_code", "centrality_overall"])
        G=nx.from_pandas_edgelist(self.trade_data, 
                          "exporter_code", "importer_code", ["import_fraction"])
        weighted_centrality = nx.eigenvector_centrality(G, weight = "import_fraction", max_iter= 10000) 
        weighted_centrality  = pd.DataFrame(list(map(list, weighted_centrality.items())), 
                                               columns = ["country_code", "weighted_centrality"])
        self.centrality_overall = pd.merge(self.centrality_overall, weighted_centrality, on = "country_code")
        
                               
        ##### COMPUTE CENTRALITY FOR COUNTRY IN PRODUCT CATEGORIES

        #sum imports across all products between countries into single value 
        imports_table_grouped = imports_table.groupby(["importer_code", "exporter_code"])["import_value"].sum().reset_index()
        products_grouped = imports_table.groupby(["product_category"])["import_value"].sum().reset_index()
        products_grouped = products_grouped.rename(columns = {"import_value": "import_product_total"})
        
        #sum exports in each category 
        self.export_types = imports_table.groupby(["importer_code", "exporter_code", "product_category"])["import_value"].sum().reset_index()
        self.export_types = pd.merge(products_grouped, self.export_types, on = "product_category")
        
        self.export_types["product_export_fraction"] = self.export_types["import_value"]\
                                                    /self.export_types["import_product_total"]*100
        
        list_products = list(set(self.export_types["product_category"]))
        
        i = 0 
        for product in list_products:
            
            temp = self.export_types[self.export_types["product_category"] == product].copy()
            
            G_w=nx.from_pandas_edgelist(temp, 
                "exporter_code", "importer_code", ["product_export_fraction"], create_using = nx.DiGraph())
            centrality_product_w = nx.eigenvector_centrality(G_w, weight = "product_export_fraction", 
                                                           max_iter= 10000)

            G=nx.from_pandas_edgelist(temp,"exporter_code", "importer_code", create_using = nx.DiGraph())
            centrality_product = nx.eigenvector_centrality(G,max_iter= 10000)

            if(i == 0):
                self.centrality_product = pd.DataFrame(list(map(list, centrality_product.items())), 
                                               columns = ["country_code", "prod_" + product])
                

            else: 
                self.centrality_product = pd.merge(self.centrality_product, 
                                               pd.DataFrame(list(map(list, centrality_product.items())), 
                                               columns = ["country_code", "prod_" + product]), 
                                                  on = "country_code")
                
            self.centrality_product = pd.merge(self.centrality_product, 
                                               pd.DataFrame(list(map(list, centrality_product_w.items())), 
                                               columns = ["country_code", "prod_w_" + product]), 
                                                  on = "country_code")
            
            i = i+1         
    
    def combine_normalize_features(self):
        
        self.combined_features = pd.merge(self.features, self.centrality_overall,on = "country_code")
        self.combined_features = pd.merge(self.combined_features, self.centrality_product,on = "country_code")
        #step eliminates NA and nodes that are not in graph, since they will have NA for graph features
        self.combined_features = self.combined_features.drop(columns = ["index"])
        #filter both trade data and features data to same subset of countries
        self.combined_features = self.combined_features[\
                                self.combined_features.country_code.isin(self.trade_data.importer_code)|\
                                self.combined_features.country_code.isin(self.trade_data.exporter_code)]
        self.trade_data = self.trade_data[\
                          self.trade_data.importer_code.isin(self.combined_features.country_code)&\
                          self.trade_data.exporter_code.isin(self.combined_features.country_code)]
        
        features_to_norm = list(self.combined_features.columns.copy())
        non_norm = ["country_code", "node_numbers"]
        cols_insufficient_data = list(self.combined_features.loc[:, self.combined_features.nunique() < 2].columns.copy())
        non_norm.extend(cols_insufficient_data)
 
        features_to_norm = [x for x in features_to_norm if x not in non_norm]
        scaler = StandardScaler()
        #we preserve NAs in the scaling
        self.combined_features[features_to_norm] = scaler.fit_transform(self.combined_features[features_to_norm])
        self.combined_features.fillna(0, inplace = True) #we fill NA after scaling 
        #check that feature has at least 20% coverage in a given year -- otherwise set to NA
        for feature in self.feature_list:
            coverage = len(self.combined_features[self.combined_features[feature] != 0])/len(self.combined_features)
            if(coverage <= 0.20): self.combined_features[feature] = 0