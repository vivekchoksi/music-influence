#!/usr/bin/python

'''
File: graph_statistics.py
=========================
Construct the influence graph using snap.py and compute graph statistics.
'''

import time
import cPickle as pickle
import snap

class GraphStatistics(object):
    # The snap directed graph object.
    Graph = None

    # The mapping from Rovicorp artist ID to snap graph node ID.
    ids = {}

    def __init__(self):
        pass

    def load_graph_from_edges(self, filename):
        print 'Loading graph from file:', filename, '...'
        self.Graph = snap.TNGraph.New()
        with open(filename, 'r') as edges:
            for line in edges:
                source, target = line.split(';')

                if source not in self.ids:
                    self.ids[source] = self.Graph.GetMxNId() + 1
                    self.Graph.AddNode(self.ids[source])

                if target not in self.ids:
                    self.ids[target] = self.Graph.GetMxNId() + 1
                    self.Graph.AddNode(self.ids[target])

                self.Graph.AddEdge(self.ids[source], self.ids[target])
        print 'Done'

    def load_graph_from_pickle(self, pickle_filename):
        print 'Loading graph from file:', pickle_filename, '...'
        self.Graph = pickle.load(open(pickle_filename, 'rb'))
        print 'Done'

    def pickle_dump_graph(self, pickle_filename):
        print 'Dumping graph to file:', pickle_filename, '...'
        pickle.dump(self.Graph, open(pickle_filename, 'wb'))
        print 'Done'

    def print_statistics(self, outfile_name):
        print 'Writing to file:', outfile_name
        snap.PrintInfo(self.Graph, 'Python type TUNGraph', outfile_name, False)

        with open(outfile_name, 'a') as f:
            f.write('\n####More information')

        max_degree_node = snap.GetMxDegNId(self.Graph)
        for artist_id in self.ids:
            if self.ids[artist_id] == max_degree_node:
                print artist_id


        # These may throw gnuplot errors; if so, edit the generated .plt files to correct the errors and run
        # gnuplot from terminal. (May need to set terminal to svg instead of png depending on your gnuplot
        # installation.)
        snap.PlotOutDegDistr(self.Graph, 'out_degree_distr', 'Out-degree distribution')
        snap.PlotInDegDistr(self.Graph, 'in_degree_distr', 'In-degree distribution')


def main():
    graph_statistics = GraphStatistics()
    graph_statistics.load_graph_from_edges('../data/song_artists_only_edges.csv')
    graph_statistics.print_statistics('reduced-graph-info.txt')

if __name__ == '__main__':
    main()
