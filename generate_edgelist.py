#!/usr/bin/python

# CS224w autumn 2015

import cPickle as pickle

def generate_edge_lists():
    influencers = pickle.load(open('influencers_graph.pickle', 'rb'))

    with open('edges.csv', 'w') as f:
        f.write('Source;Target\n')
        for artist_id in influencers:
            for influencer_id in influencers[artist_id]:
                f.write(str(influencer_id) + ';' + str(artist_id) + '\n')

def generate_node_labels():
    artist_ids = pickle.load(open('artist_ids.pickle', 'rb'))

    with open('node_labels.csv', 'w') as f:
        f.write('Id;Label\n')
        for artist_id in artist_ids:
            f.write(artist_id.encode('utf-8') + ';' + artist_ids[artist_id].encode('utf-8') + '\n')


def main():
    generate_edge_lists()
    generate_node_labels()

if __name__ == '__main__':
    main()