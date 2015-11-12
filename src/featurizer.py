import os
import pandas as pd
from networkx import Graph
from loader import GraphLoader
import logging

class FeatureGenerator(object):
    """
    Generates learning features from the influence graph
    """
    basepath = None
    IG = None  # All Music Influence Graph
    sdf = None  # Song DataFrame with audio features

    def __init__(self, IG, verbose=True):
        """
        :param IG: All music influence graph, type networkx.Graph
        """
        self.verbose = verbose
        self.basepath = os.path.dirname(os.path.dirname(__file__))
        self.sdf = GraphLoader().load_song_dataframe()
        self.IG = IG

    def log(self, *args, **kwargs):
        if self.verbose: logging.info(*args, **kwargs)


    def get_features(self, u, v):
        """
        :return: dict of features for influence edge (u,v)
        """
        return {"common_neighbors": self._ncommon_neighbors(u,v)}

    def feature_matrix(self, edges):
        """
        :param edges: list of (u,v) tuples representing nodes in the influence graph
        :return: feature matrix where each row is the feature mapping of edges to features
        """
        # Features: As baseline just doing # of common neighbors
        # Other possible features, edge closeness centrality, Lada-Adamic
        features = []
        percent = int(10*len(edges))
        self.log("Generating feature matrix for {} edges".format(len(edges)))
        for i, (u,v) in enumerate(edges):
            if i % percent == 0: self.log("...{}% progress".format(i%percent*10))
            features.append(self.get_features(u,v))
        return features


    def _ncommon_neighbors(self, u, v):
        return len(set(self.IG.neighbors(v)) | set(self.IG.neighbors(u)))


if __name__ == '__main__':
    IG = GraphLoader().load_networkx_influence_graph(pruned=False)
    featurizer = FeatureGenerator(IG)


