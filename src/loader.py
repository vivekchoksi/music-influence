#!/usr/bin/python

'''
File: loader.py
=========================
Construct the influence graph using snap.py, or networkx and compute graph statistics.
'''

import time
import cPickle as pickle
import networkx as nx
import codecs
import snap
import os
import logging


class GraphLoader(object):
    """ Graph Loader class to facilitate quickly loading snap/networkx influence/song graphs
    """
    def __init__(self, path=None, verbose=False):
        self.basepath = os.path.dirname(os.path.dirname(__file__)) if path is None else path
        self.verbose = verbose

    def _log(self, *args, **kwargs):
        if self.verbose:
            logging.info(*args, **kwargs)

    def load_snap_influence_graph(self, path=None, from_edge_file=True):
        if from_edge_file:
            return self._load_snap_influence_from_edge_file(path)
        else:
            return self._load_snap_influence_from_pickle_file(path)

    def _load_snap_influence_from_pickle_file(self, path=None):
        filepath = os.path.join(self.basepath, "data", "influencers_graph.pickle") if path is None else path
        self._log('Loading graph from file:{}...'.format(filepath))
        Graph = pickle.load(open(path, 'rb'))
        self._log('Done')
        return Graph

    def _load_snap_influence_from_edge_file(self, path=None):
        """
        :return: snap.TNGraph read from default path, or absolute path if passed in
        """
        filepath = os.path.join(self.basepath, "data", "edges.csv") if path is None else path
        self._log('Loading graph from file:{}...'.format(filepath))
        Graph = snap.TNGraph.New()
        ids = {}
        with open(filepath, 'r') as edges:
            for line in edges:
                source, target = line.split(';')

                if source not in ids:
                    ids[source] = Graph.GetMxNId() + 1
                    Graph.AddNode(ids[source])

                if target not in ids:
                    ids[target] = Graph.GetMxNId() + 1
                    Graph.AddNode(ids[target])

                Graph.AddEdge(ids[source], ids[target])
        self._log('Done')
        return Graph, ids

    def pickle_dump_graph(self, Graph, pickle_filename):
        self._log('Dumping graph to file:{}...'.format(pickle_filename))
        pickle.dump(Graph, open(pickle_filename, 'wb'))
        self._log('Done')

    def load_networx_influence_graph(self, path=None):
        """
        :return: networkx influence graph there id is rovicorp id and each node has
        the artist name as one of its attributes
        """
        filepath = os.path.join(self.basepath, "data", "edges.csv") if path is None else path
        G = nx.read_edgelist(os.path.join(filepath, 'edges.csv'), delimiter=';', comments="Source")
        names = self.get_artist_ids_to_names()
        for nid in G.node:
            if nid in names:
                G.node[nid].update(artist_name=names[nid])
        return

    def get_artist_ids_to_names(self, path=None):
        """
        :return: Returns dict from rovicorp artist ids to names
        """
        filepath = os.path.join(self.basepath, "data", "node_labels.csv") if path is None else path
        names = {}  # Rovicorp ids => names
        with codecs.open(filepath, encoding="utf-8") as fin:
            for line in fin.readlines():
                nid, name = line.split(";")
                if nid == "Id": continue  # Header
                names[nid] = name
        return names

def main():
    ldr = GraphLoader()
    G = ldr.load_snap_influence_graph()  # snap TNGraph
    Gnx = ldr.load_networx_influence_graph()  # networkx influence graph

if __name__ == '__main__':
    main()
