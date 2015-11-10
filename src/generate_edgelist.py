#!/usr/bin/python

'''
File: generate_edgelist.py
==========================
Load graph data from pickle files and output as CSV for input into Gephi.
'''

import cPickle as pickle

def generate_edge_lists(influence_graph):
    influencers = pickle.load(open(influence_graph, 'rb'))

    with open('../data/edges.csv', 'w') as f:
        f.write('Source;Target\n')
        for artist_id in influencers:
            for influencer_id in influencers[artist_id]:
                f.write(str(influencer_id) + ';' + str(artist_id) + '\n')

def generate_node_labels(artist_ids):
    artist_ids = pickle.load(open(artist_ids, 'rb'))

    with open('../data/node_labels.csv', 'w') as f:
        f.write('Id;Label\n')
        for artist_id in artist_ids:
            f.write(artist_id.encode('utf-8') + ';' + artist_ids[artist_id].encode('utf-8') + '\n')

def main():
    generate_edge_lists('../data/influencers_graph.pickle')
    generate_node_labels('../data/artist_ids.pickle')

if __name__ == '__main__':
    main()
