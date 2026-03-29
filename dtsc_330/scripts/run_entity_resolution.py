import networkx as nx
import pandas as pd
import sqlalchemy

from dtsc330_26 import entity_resolution_features, grantee_nn_idx
from dtsc330_26.classifiers import entity_resolution_classifier
from dtsc330_26.readers import articles, grants

eres_features = entity_resolution_features.EntityResolutionFeatures()
classifier = entity_resolution_classifier.EntityResolutionClassifier()
nnidx = grantee_nn_idx.GranteeNNIdx()

# 0. Train our classifier
# Load the training data
df = (
    pd.read_csv("data/temp-auth-grantee-training-10.csv")[
        ["a_forename", "g_forename", "a_affiliation", "g_department", "is_match"]
    ]
    .reset_index(drop=True)
    .dropna()
)
labels = df["is_match"]
df = df[["a_forename", "g_forename", "a_affiliation", "g_department"]]
df = df.rename(
    columns={
        "a_forename": "forename_x",
        "g_forename": "forename_y",
        "a_affiliation": "affiliation_x",
        "g_department": "affiliation_y",
    }
)

# Compute distances
features = eres_features.features(df)
# Train
classifier.train(features, labels)

engine = sqlalchemy.create_engine("sqlite:///data/article_grant_db.sqlite")
connection = engine.connect()

# 1. Read in batches of data
# We'll loop through authors (larger dataset) and compare against grantees
art = articles.Articles("data/pubmed25n1275.xml.gz")
for chunk in art.batch_from_db():
    # Everything is within a batch/block
    for _, row in chunk.iterrows():
        # row is AUTHOR id, forename, surname, affiliation (by 1)
        nns, dists = nnidx.query(row["forename"], row["surname"], row["affiliation"])
        # nns are GRANTEES id, forename, surname, affiliation (by 100 nns)

        comb_df = nns.rename(
            columns={"forename": "forename_y", "affiliation": "affiliation_y"}
        )[["forename_y", "affiliation_y"]]
        comb_df["forename_x"] = row["forename"]
        comb_df["affiliation_x"] = row["affiliation"]

        # 2. Compute distances within this block
        features = eres_features.features(comb_df)
        # 3. Run through classifier to figure out matches
        labels = classifier.predict(features) > 0.5

        matches = nns[labels]
        if len(matches) > 0:
            matches["author_id"] = row["id"]
            matches["grantee_id"] = matches["id"]
            # 4. Store matches
            matches[["author_id", "grantee_id"]].to_sql("grantee_author", connection)

# When completed
# Find overlaps using graph theory (connected components)

df = pd.read_sql("SELECT grantee_id, author_id FROM grantee_author", connection)
gx = nx.Graph()
# Not directed graph (DiGraph) -- connections go both ways just like
# facebook friends

# Need to add nodes and edges
# grantee/author id is node
# edge is a connection between the two where we found a match

df["grantee_id"] = df["grantee_id"].apply(lambda x: "g" + str(x))
df["author_id"] = df["author_id"].apply(lambda x: "a" + str(x))
df["edge"] = df.apply(lambda row: (row["grantee_id"], row["author_id"]))
nodes = df["grantee_id"].to_list() + df["author_id"].to_list()
edges = df["edge"].to_list()

gx.add_nodes_from(nodes)
gx.add_edges_from(edges)

people = gx.connected_components()

# Create our people representation from taht
