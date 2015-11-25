#!/usr/bin/python

'''
File: loader.py
=========================
Construct the influence graph using snap.py, or networkx and compute graph statistics.
'''

import time
import cPickle as pickle
import networkx as nx
import pandas as pd
import codecs
import snap
import os
import logging
logging.basicConfig(format="[%(name)s %(asctime)s]\t%(msg)s", level=logging.INFO)


class GraphLoader(object):
    """ Graph Loader class to facilitate quickly loading snap/networkx influence/song graphs
    """
    def __init__(self, path=None, verbose=True):
        self.basepath = os.path.dirname(os.path.dirname(__file__)) if path is None else path
        self.verbose = verbose

    def log(self, *args, **kwargs):
        if self.verbose:
            logging.info(*args, **kwargs)

    def load_snap_influence_graph(self, path=None, from_edge_file=True):
        if from_edge_file:
            return self._load_snap_influence_from_edge_file(path)
        else:
            return self._load_snap_influence_from_pickle_file(path)

    def _load_snap_influence_from_pickle_file(self, path=None):
        filepath = os.path.join(self.basepath, "data", "influencers_graph.pickle") if path is None else path
        self.log('Loading graph from file:{}...'.format(filepath))
        Graph = pickle.load(open(path, 'rb'))
        self.log('Done')
        return Graph

    def _load_snap_influence_from_edge_file(self, path=None):
        """
        :return: snap.TNGraph read from default path, or absolute path if passed in
        """
        filepath = os.path.join(self.basepath, "data", "song_artists_only_edges.csv") if path is None else path
        self.log('Loading graph from file:{}...'.format(filepath))
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
        self.log('Done')
        return Graph, ids

    def pickle_dump_graph(self, Graph, pickle_filename):
        self.log('Dumping graph to file:{}...'.format(pickle_filename))
        pickle.dump(Graph, open(pickle_filename, 'wb'))
        self.log('Done')

    def load_networkx_influence_graph(self, path=None, pruned=True):
        """
        :return: networkx influence graph there id is rovicorp id and each node has
        the artist name as one of its attributes
        """
        filepath = os.path.join(self.basepath, "data", "song_artists_only_edges.csv") if path is None else path
        G = nx.read_edgelist(filepath, delimiter=';', comments="Source")
        names = self.get_artist_ids_to_names()
        for nid in G.node:
            if nid in names:
                G.node[nid].update(artist_name=names[nid].lower())

        return self.prune_influence_graph(G) if pruned else G

    def load_song_dataframe(self, path=None):
        file_path = os.path.join(self.basepath, "data", "evolution.csv") if path is None else path
        return pd.read_csv(file_path)

    def prune_influence_graph(self, IG, path=None):
        """
        :return: A pruned copy of the influence graph passed in that
        only keeps artists' whose songs exist in music graph
        """
        sdf = self.load_song_dataframe(path)
        IGpruned = IG.copy()
        names_in_song_data =  [st.lower() for st in set(sdf["artist_name"].values.tolist())] # artist names in song graph
        for nid in IG.node:
            if "artist_name" not in IG.node[nid]:
                self.log(u"Artist name was never set for artist: {}".format(nid))
                continue
            name = IG.node[nid]["artist_name"]
            if name not in names_in_song_data:
                self.log(u"Artist: {} id: {} did not have any songs in song dataset".format(name.strip(), nid))
                del IGpruned.node[nid]  # Delete this artist from the pruned graph
            else:
                self.log(u"Artist: {} id: {} had a song in song dataset".format(name, nid))

        return IGpruned

    def get_artist_ids_to_names(self, path=None):
        """
        :return: Returns dict from rovicorp artist ids to names
        """
        filepath = os.path.join(self.basepath, "data", "song_artists_only_labels.csv") if path is None else path
        names = {}  # Rovicorp ids => names
        with codecs.open(filepath, encoding="utf-8") as fin:
            for line in fin.readlines():
                ls = line.split(";")
                if len(ls) != 2:
                    self.log(u"Badly formated line: {}".format(line))
                    continue
                nid, name = ls
                if nid == "Id": continue  # Header
                names[nid] = name
        return names

def main():
    ldr = GraphLoader()
    G = ldr.load_snap_influence_graph()  # snap TNGraph
    Gnx = ldr.load_networkx_influence_graph()  # networkx influence graph

if __name__ == '__main__':
    main()
