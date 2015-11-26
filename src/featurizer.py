import os
import pandas as pd
from networkx import Graph
import networkx as nx
import sys
from networkx.exception import NetworkXNoPath
from loader import GraphLoader
import random
import math
import logging
import pickle

class FeatureGenerator(object):
    """
    Generates learning features from the influence graph
    """
    basepath = None
    IG = None  # All Music Influence Graph
    sdf = None  # Song DataFrame with audio features
    genres = {}
    time_active = {}

    def __init__(self, verbose=True, features_to_use=None):
        """
        :param IG: All music influence graph, type networkx.Graph
        """
        self.verbose = verbose
        self.basepath = os.path.dirname(os.path.dirname(__file__))
        self.sdf = GraphLoader().load_song_dataframe()
        self.IG = None
        features_to_use = ["nc"] if features_to_use is None else features_to_use
        self.feature_mappers = self.get_feature_mappers(features_to_use)
        self.log("Featurizer will use: {}".format(self.get_feature_names()))
        self.load_genres()
        self.load_time_active()

    def log(self, *args, **kwargs):
        if self.verbose: logging.info(*args, **kwargs)

    def set_graph(self, IG):
        self.IG = IG
        self.UIG = IG.to_undirected()

    def load_genres(self):
        self.genres = pickle.load(open('../data/genres.pickle', 'rb'))

    def load_time_active(self):
        self.time_active = pickle.load(open('../data/time_active.pickle', 'rb'))

    def get_feature_names(self):
        names, funcs = zip(*self.feature_mappers)
        return names

    def get_feature_mappers(self, features_to_use):
        abbreviation_map = {
            "nc": "ncommon_neighbors",
            "jc": "jaccard_coefficient",
            "aa": "adamic_adar",
            "pa": "preferential_attachment",
            "pp": "personalized_page_rank",
            #"sp": "len_shortest_undirected_path",
            "ra": "resource_allocation",
            "si": "sorensen_index",
            "lh": "leicht_holme_newman",
            "g": "genres", 
            "ta":"time_active"

        }
        feature_mappers = {
            "ncommon_neighbors": self._ncommon_neighbors,
            "jaccard_coefficient": self._jaccard_coeff,
            "adamic_adar": self._adamic_adar,
            "preferential_attachment": self._preferential_attachment,
            "personalized_page_rank": self._ppage_rank,
            "len_shortest_undirected_path": self._len_shortest_path,
            "resource_allocation": self._resource_allocation,
            "sorensen_index": self._sorensen_index,
            "leicht_holme_newman": self._leicht_holme_newman,
            "genres": self._genre, 
            "time_active":self._time_active
        }
        result = []
        for abbrv in features_to_use:
            if abbrv in abbreviation_map.keys():
                feature_name = abbreviation_map[abbrv]
                result.append( (feature_name, feature_mappers[feature_name]) )
        return result


    def compute_features(self, u, v):
        """
        :return: list of feature mappings for influence edge (u,v)
        """
        edge_features = [func(u,v) for name, func in self.feature_mappers]
        return edge_features

    def _sorensen_index(self, u, v):
        if (float(self.IG.degree(u) + self.IG.degree(v))) == 0: return 0
        return self._ncommon_neighbors(u,v) / float(self.IG.degree(u) + self.IG.degree(v))

    def _leicht_holme_newman(self, u, v):
        if (self.IG.degree(u) * self.IG.degree(v)) == 0: return 0
        return self._ncommon_neighbors(u,v) / float(self.IG.degree(u) * self.IG.degree(v))

    def _common_neighbors(self, u, v):
        return set(self.IG.neighbors(v)) & set(self.IG.neighbors(u))

    def _ncommon_neighbors(self, u, v):
        return len(self._common_neighbors(u, v))

    def _jaccard_coeff(self, u, v):
        return float(self._ncommon_neighbors(u,v)) / (len(set(self.IG.neighbors(v)) | set(self.IG.neighbors(u))))

    def _adamic_adar(self, u, v):
        return sum([ 1.0 / math.log(self.IG.degree(z)) for z in self._common_neighbors(u,v)])

    def _resource_allocation(self, u, v):
        return sum([ 1.0 / self.IG.degree(z) for z in self._common_neighbors(u,v)])

    def _preferential_attachment(self, u, v):
        return self.IG.degree(u) * self.IG.degree(v)

    def _random(self, u, v):
        return 1 if random.uniform(0,1) <=.5 else 0

    def _genre(self, u, v):
        if u not in self.genres or v not in self.genres:
            return 0
        else:
            genre_u = self.genres[u]
            genre_v = self.genres[v]
            return float(len(set(genre_u).intersection(set(genre_v)))/len(set(genre_u).union(set(genre_v))))

    def _time_active(self, u, v):
        if u not in self.time_active or v not in self.time_active:
            return 0
        else:
            time_u = self.genres[u]
            time_v = self.genres[v]
            return float(len(set(time_u).intersection(set(time_v)))/len(set(time_u).union(set(time_v))))

    def _len_shortest_path(self, u, v):
        try:
            path = nx.shortest_path(self.UIG, source=u, target=v)
        except NetworkXNoPath, e:
            self.log("\tNo path between {} and {}, will set to graph distance to sys.maxint, this will most likely"
                    " lead to garbage for any classifier".format(u, v))
            return sys.maxint - 2
        return len(path)-2

    def _ppage_rank(self, u, v):
        personal = {nid: 0 for nid in self.IG.node}
        personal[u] = 1.0
        r_uv = nx.pagerank(self.IG, personalization=personal).get(v)
        personal[u] = 0
        personal[v] = 1.0
        r_vu = nx.pagerank(self.IG, personalization=personal).get(u)
        return r_uv + r_vu

    def feature_matrix(self, edges):
        """
        :param edges: list of (u,v) tuples representing nodes in the influence graph
        :return: feature matrix where each row is the feature mapping of edges to features
        """
        # Features: As baseline just doing # of common neighbors
        # Other possible features, edge closeness centrality, Lada-Adamic
        features = []
        percent = int(len(edges)/10)
        self.log("Generating feature matrix for {} edges".format(len(edges)))
        for i, (u,v) in enumerate(edges):
            #if i % percent == 0: self.log("\t...{}% progress".format((i/percent)*10))
            features.append(self.compute_features(u,v))
        self.log("done")
        return features



if __name__ == '__main__':
    IG = GraphLoader().load_networkx_influence_graph(pruned=False)
    featurizer = FeatureGenerator(IG)


