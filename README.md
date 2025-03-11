# Trade-GNN-Thesis
Primary research objective: Can graph machine learning be used to better predict economic shocks in the international trade network (ITN) compared to traditional change-point detection methods?
Secondary research objective: How do the detected economic shocks differ by region and product sub-networks?

## Overview

This project explores the use of **Siamese Graph Neural Networks (Siamese-GNNs)** for detecting change-points in economic systems, specifically within the context of international trade networks. By leveraging graph-based approaches, the model analyzes shifts in the underlying structure of the trade network and identifies economic shocks that impact trade flows. This work is grounded in both network analysis and time series modeling, providing insights into the dynamics of global trade during times of economic turmoil.
![Trade Network](Images/world_network.png)

## Key Concepts

- **Siamese Networks:** A type of neural network architecture used to determine whether two input samples are similar or not. In this project, the Siamese network is extended to graph-structured data.
  
- **Graph Neural Networks (GNNs):** A class of neural networks designed to process data represented as graphs. GNNs can capture complex relationships in networks by passing messages between nodes and aggregating information from their neighbors.
  
- **Change-Point Detection:** A method of identifying points in time where the properties of a time series change. In this case, it identifies shifts in the international trade network that could represent economic shocks.

- **Economic Shocks:** Unexpected events that have a significant and lasting impact on economic systems, such as financial crises or trade policy changes.
![Trade Network](Images/Global_Crises.png)

## Project Goals

- Develop a Siamese-GNN architecture to detect change-points in the international trade network.
- Analyze how economic shocks affect the trade network structure.
- Investigate how the trade relationships between countries change over time due to these shocks.

## Approach

1. **Data Collection and Preprocessing:**
   - Data was collected from publicly available trade databases, focusing on bilateral trade data between countries over several years.
   - The data is represented as graphs, where nodes represent countries, and edges represent trade relationships with weights corresponding to the volume of trade.
   - Preprocessing includes normalizing trade data, handling missing values, and constructing dynamic graphs that evolve over time.

2. **Model Design:**
   - The Siamese-GNN model was designed to compare two graphs from different time periods and determine if they belong to the same period or if a change-point has occurred.
   - The network utilizes GNN layers to encode graph structures and Siamese network architecture to compare two graphs by computing a similarity score.
   - A custom loss function was designed to minimize the difference between the embeddings of similar graphs and maximize the difference for dissimilar graphs.
   ![Trade Network](Images/model_layers.png)

3. **Training and Evaluation:**
   - The model was trained using historical trade data, with the true change-points labeled using domain knowledge of global economic shocks.
   - Pairs of networks are labeled as 0 if they are separated by an economic shock (not belonging to the same distribution) or 1 if they are not (belong to same distribution).
   - Performance was evaluated using standard metrics such as precision, recall, and F1 score, with a focus on minimizing false positives (i.e., incorrectly detecting a change-point) and false negatives (i.e., missing a true change-point).

4. **Analysis:**
   - Subnetworks of regions and products were then analyzed to understand how these are differentially impacted by economic shocks, with significant differences found.

## Folders and Files
1. **SRC** 
- Contains train.py which can be run to train the model contained in model.py
- Utils: helper functions for creating training pairs, detecting change-points after training models, traditional CPD methods, etc.
2. **Notebooks**
- Jupyter notebooks used to create networks with export data and MIS/random subsets of World Bank features
