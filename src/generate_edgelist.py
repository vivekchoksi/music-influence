#!/usr/bin/python

'''
File: generate_edgelist.py
==========================
Load graph data from pickle files and output as CSV for input into Gephi.
'''

import cPickle as pickle

def generate_edge_lists(influence_graph, file_to_write):
    influencers = pickle.load(open(influence_graph, 'rb'))

    with open('../data/%s.csv' %file_to_write, 'w') as f:
        f.write('Source;Target\n')
        for artist_id in influencers:
            for influencer_id in influencers[artist_id]:
                f.write(str(influencer_id) + ';' + str(artist_id) + '\n')

def generate_node_labels(artist_ids, file_to_write, is_first=False):
    artist_ids = pickle.load(open(artist_ids, 'rb'))

    with open('../data/%s.csv' %file_to_write, 'a') as f:
        f.write('Id;Label\n')
        for artist_id in artist_ids:
            f.write(artist_id.encode('utf-8') + ';' + artist_ids[artist_id].encode('utf-8') + '\n')

def generate_artist_songs(artist_to_songs, file_to_write):
    song_indices = pickle.load(open(artist_to_songs, 'rb'))

    with open('../data/%s.csv' %file_to_write, 'w') as f:
        f.write('Id;Song row indices\n')
        for artist_id in song_indices:
            f.write(str(artist_id) + ';' + str(song_indices[artist_id]) + '\n')

def generate_genres(genres, file_to_write):
    artist_genres = pickle.load(open(genres, 'rb'))

    with open('../data/%s.csv' %file_to_write, 'w') as f:
        f.write('Id;genres\n')
        for artist_id in artist_genres:
            f.write(str(artist_id) + ';' + str(artist_genres[artist_id]) + '\n')    
            
def generate_time_active(times, file_to_write):
    artist_times = pickle.load(open(times, 'rb'))

    with open('../data/%s.csv' %file_to_write, 'w') as f:
        f.write('Id;times active\n')
        for artist_id in artist_times:
            f.write(str(artist_id) + ';' + str(artist_times[artist_id]) + '\n')            

def main():
    generate_edge_lists('../data/influencers_graph.pickle')
    generate_node_labels('../data/artist_ids.pickle')

if __name__ == '__main__':
    main()
