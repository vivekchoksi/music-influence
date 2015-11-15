import os
import pandas as pd
from networkx import Graph
import networkx as nx
from loader import GraphLoader
import random
import math
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


    def get_feature_names(self):
        cm = "ncommon_neighbors"
        jc = "jaccard_coefficient"
        aa = "adamic_adar"
        pa = "preferential_attachment"
        pp = "personalized page rank"
        return [pp]

    def get_features(self, u, v):
        """
        :return: list of feature mappings for influence edge (u,v)
        Other scoring functions per Jure's lecture:
        Jaccard coeff:   ncommon_neighbors / union_neighbors
        Graph Distaince: negated shortest path length
        Adamic/Adar: sum_{z \in common_neighbors} 1 / log(deg(z))
        Preferential Attachment:  deg(u) * deg(v)
        PPageRank:    r_u{v} + r_v{u}
        """
        rd = self._random(u,v)
        #cn = self._ncommon_neighbors(u,v)
        #jc = self._jaccard_coeff(u,v)
        #aa = self._adamic_adar(u,v)
        #pa = self._preferential_attachment(u,v)
        #pp = self._ppage_rank(u,v)  # not tractable also not good according to Jure
        ra = self._resource_allocation(u,v)
        sp = self._len_shortest_path(u, v)
        return [sp]

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

    def _len_shortest_path(self, u, v):
        return len(nx.shortest_path(self.IG, source=u, target=v))-2

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
            if i % percent == 0: self.log("\t...{}% progress".format((i/percent)*10))
            features.append(self.get_features(u,v))
        self.log("done")
        return features



if __name__ == '__main__':
    IG = GraphLoader().load_networkx_influence_graph(pruned=False)
    featurizer = FeatureGenerator(IG)


