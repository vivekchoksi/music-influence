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
import ast
import logging
import pdb
logging.basicConfig(format="[%(name)s %(asctime)s]\t%(msg)s", level=logging.INFO)


class GraphLoader(object):
    """ Graph Loader class to facilitate quickly loading snap/networkx influence/song graphs
    """

    DATA_DIR = "data"
    EDGES_PICKLE = "song_artists_edges_only.pickle"
    EDGES_FILENAME = "song_artists_only_edges.csv"
    EVOLUTION_FILENAME = "evolution.csv"
    LABELS_FILENAME = "song_artists_only_labels.csv"
    SONG_VECTORS_FILENAME = "song_vectors.csv"
    SONG_VECTORS_PICKLE = "song_vectors.pickle"
    ARTISTS_TO_YEARS_PICKLE = 'artists_to_years.pickle'


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
        filepath = os.path.join(self.basepath, self.DATA_DIR, self.EDGES_PICKLE) if path is None else path
        self.log('Loading graph from file:{}...'.format(filepath))
        Graph = pickle.load(open(path, 'rb'))
        self.log('Done')
        return Graph

    def _load_snap_influence_from_edge_file(self, path=None):
        """
        :return: snap.TNGraph read from default path, or absolute path if passed in
        """
        filepath = os.path.join(self.basepath, self.DATA_DIR, self.EDGES_FILENAME) if path is None else path
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

    def pickle_dump_song_vectors(self, path=None, outfile_path=None):
        """
        Pickle dump a dictionary mapping from artist ID to list of lists of audio features.
        """
        filepath = os.path.join(self.basepath, self.DATA_DIR, self.SONG_VECTORS_FILENAME) if path is None else path
        outfile_path = os.path.join(self.basepath, self.DATA_DIR, self.SONG_VECTORS_PICKLE) \
            if outfile_path is None else outfile_path
        self.log('Dumping song vectors to file:{}...'.format(outfile_path))

        song_vectors = {}
        with open(filepath, 'r') as f:
            for line in f:
                artist_id, vectors_as_string = line.split(';')
                vectors = ast.literal_eval(vectors_as_string)
                for idx, vector in enumerate(vectors):
                    vectors[idx] = [float(val) for val in vector]
                song_vectors[artist_id] = vectors
        pickle.dump(song_vectors, open(outfile_path, 'wb'))
        self.log('Done')

    def load_song_vectors(self, path=None):
        """
        :return: song vectors loaded from a pickle file, mapping rom artist ID to list of lists of audio features
        """
        filepath = os.path.join(self.basepath, self.DATA_DIR, self.SONG_VECTORS_PICKLE) if path is None else path
        self.log('Loading song data from file:{}...'.format(filepath))
        song_vectors = pickle.load(open(filepath, 'rb'))
        self.log('Done')
        return song_vectors

    def load_artists_to_years(self, path=None):
        """
        :return: dictionary mapping rom artist ID to list of years when the artist's songs were released
        """
        filepath = os.path.join(self.basepath, self.DATA_DIR, self.ARTISTS_TO_YEARS_PICKLE) if path is None else path
        self.log('Loading artist years data from file:{}...'.format(filepath))
        artists_to_years = pickle.load(open(filepath, 'rb'))
        self.log('Done')
        return artists_to_years  

    def pickle_dump_graph(self, Graph, pickle_filename):
        self.log('Dumping graph to file:{}...'.format(pickle_filename))
        pickle.dump(Graph, open(pickle_filename, 'wb'))
        self.log('Done')

    def load_networkx_influence_graph(self, path=None, pruned=True):
        """
        :return: networkx influence graph there id is rovicorp id and each node has
        the artist name as one of its attributes
        """
        filepath = os.path.join(self.basepath, self.DATA_DIR, self.EDGES_FILENAME) if path is None else path
        G = nx.read_edgelist(filepath, delimiter=';', create_using=nx.DiGraph(), comments="Source")
        names = self.get_artist_ids_to_names()
        for nid in G.node:
            if nid in names:
                G.node[nid].update(artist_name=names[nid].lower())

        return self.prune_influence_graph(G) if pruned else G

    def load_song_dataframe(self, path=None):
        file_path = os.path.join(self.basepath, self.DATA_DIR, self.EVOLUTION_FILENAME) if path is None else path
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
        filepath = os.path.join(self.basepath, self.DATA_DIR, self.LABELS_FILENAME) if path is None else path
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
    ldr.pickle_dump_song_vectors()

if __name__ == '__main__':
    main()
