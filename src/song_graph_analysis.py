import os
import pandas as pd
import logging

class EvolutionGraph(object):
    basepath = None
    IG = None  # All Music Influence Graph
    sdf = None  # Song DataFrame with audio features

    def __init__(self, IG, verbose=True):
        """
        :param IG: All music influence graph, type networkx.Graph
        """
        self.verbose = verbose
        self.basepath = os.path.dirname(os.path.dirname(__file__))
        file_path = os.path.join(os.path.join(self.basepath, "data"), "evolution.csv")
        self.sdf = pd.read_csv(file_path)
        self.IG = self._prune_influence_graph(IG)
        # Prune influence graph based on music feature songs we have

    def log(self, *args, **kwargs):
        if self.verbose: logging.info(*args, **kwargs)

    def _prune_influence_graph(self, IG):
        """
        :return: A pruned copy of the influence graph passed in that
        only keeps artists' whose songs exist in music graph
        """
        IGpruned = IG.copy()
        names_in_song_data = self.sdf["artist_name"] # artist names in song graph
        for nid in IG.node:
            name = IG.node[nid]["artist_name"]
            if name not in names_in_song_data:
                self.log("Artist: {} id: {} did not have any songs in song dataset".format(name, id))
                del IG.node[nid]  # Delete this artist from the pruned graph
        return IGpruned




