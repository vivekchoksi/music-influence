#!/usr/bin/python

'''
File: allmusic_scraper.py
=========================
Scrape allmusic.com's data using the Rovicorp API, generate an influence graph, and dump the graph to a pickle file.
'''

import time
import cPickle as pickle
import urllib2
import requests
import time
import hashlib
from collections import deque
import pdb
import time
import secrets

class MaxQpsExceededError(Exception):
    pass

class Scraper(object):
    API_URL = 'http://api.rovicorp.com/data/v1.1/name/'

    API_URL_SONG = 'http://api.rovicorp.com/data/v1.1/song/'

    # Credentials for the request to Rovicorp's API.
    KEY = secrets.API_KEY
    SECRET = secrets.SHARED_SECRET

    # The interval, in seconds, after which the signature should be recalculated in self._get_sig().
    REFRESH_PERIOD = 100

    # The amount of time, in seconds, to wait after issuing each API request. This rate limit is imposed to avoid
    # exceeding the maximum allowable 5 queries per second.
    WAIT_TIME = 0.2

    # The timestamp at which the signature was last updated.
    last_timestamp = None

    # The signature used to authenticate Rovi API requests.
    sig = None

    # Mapping from artist id to name; e.g. artist_ids['MN0000855531'] = 'Barbra Streisand'.
    artist_ids = {}

    # Mapping from artist id to set of ids of influencers of that artist.
    influencers= {}

    # Set of ids of artists who's influencers and followers have already been mapped.
    visited = set()

    # Queue used in breadth-first search.
    queue = None

    def __init__(self):
        self.last_timestamp = int(time.time())

    def generate_influence_graph(self, seed_id, seed_name, max_artists=None):
        '''
        Given a seed artist, perform breadth-first search on the influence graph.

        :param seed_id: id of artist at which to start search
        :param seed_name: name of artist at which to start search
        :param max_artists: number of artists for whom to fetch influencers and followers; if None, let search execute
            exhaustively
        '''
        if self.queue is None:
            self.queue = deque([seed_id])
            self.artist_ids[seed_id] = seed_name

        while len(self.queue) > 0 or (max_artists is not None and len(self.visited) < max_artists):
            artist = self.queue.pop()
            if artist not in self.visited:
                self.visited.add(artist)
                self._get_influencers(artist)
                self._get_followers(artist)

    def print_stats(self):
        '''
        Print statistics about how many artists have been scraped.
        '''
        print 'Number of artists visited:', len(self.visited)
        print 'Number of nodes in graph:', len(self.artist_ids)
        if self.queue is not None:
            print 'Number of artists in queue:', len(self.queue)

    def _get_influencers(self, artist):
        '''
        Given an artist ID, add edges to the influence graph describing each of the artist's influences.
        '''
        url = self._get_influencers_url(artist)
        response = self._get_response(url)

        if response == -1:
            # Error.
            return
        else:
            influencers_response = response['influencers']

        #print self.artist_ids[artist], 'has', len(influencers_response), 'influencers'
        for influencer in influencers_response:
            # Map influencer id to name.
            #if influencer['id'] not in self.artist_ids:
                #self.artist_ids[influencer['id']] = influencer['name']

            self._create_influence_edge(influencer['id'], artist)

    def _get_followers(self, artist):
        '''
        Given an artist ID, add edges to the influence graph describing each of the artist's followers.
        '''
        url = self._get_followers_url(artist)
        response = self._get_response(url)

        if response == -1:
            # Error.
            return
        else:
            followers_response = response['followers']

        #print self.artist_ids[artist], 'has', len(followers_response), 'followers'

        for follower in followers_response:
            # Map follower id to name.
            #if follower['id'] not in self.artist_ids:
                #self.artist_ids[follower['id']] = follower['name']
            self._create_influence_edge(artist, follower['id'])

    def get_info(self, artist_name_id):
        '''
        :return: json respones from Name/info
        '''
        return self._get_response(self._get_info_url(artist_name_id))


    def _get_info_url(self, artist_name_id):
        return self.API_URL + 'info?apikey=%s&sig=%s&nameid=%s' % (self.KEY, self._get_sig(), artist_name_id)

    def get_song(self, song_name):
        response = self._get_response(self._get_song_url(song_name))
        if response == -1:
            return -1
        else:
            return response

    def _get_song_url(self, song_name):
        return self.API_URL_SONG + 'info?apikey=%s&sig=%s&track=%s' %(self.KEY,self._get_sig(), song_name)

    def get_artist(self, artist_name):
        response = self._get_response(self._get_artist_url(artist_name))
        if response == -1:
            return -1
        else:
            return response

    def _get_artist_url(self, artist_name):
        return self.API_URL + 'info?apikey=%s&sig=%s&name=%s' % (self.KEY, self._get_sig(), artist_name)

    def _get_response(self, url):
        '''
        Return the contents of a GET request to the specified url as JSON; return -1 on error.
        '''
        resp = requests.get(url)

        # Rate limit to avoid exceeding max allowable QPS.
        time.sleep(self.WAIT_TIME)

        if resp.status_code != 200:
            print 'No influencers or followers'

            if resp.content.find('Developer Over Rate') != -1:
                raise MaxQpsExceededError
            elif resp.content.find('Not Found') != -1:
                print 'No data received from request', url

            return -1

        # Get json list of followers.
        return resp.json()

    def _create_influence_edge(self, influencer, influenced):
        '''
        Add an edge to the influence graph between the artist IDs `influencer` and `influenced`.
        '''
        # Add influencer to graph.
        if influenced not in self.influencers:
            self.influencers[influenced] = set()
        self.influencers[influenced].add(influencer)

        #if influencer not in self.visited:
           # self.queue.appendleft(influencer)
        #if influenced not in self.visited:
            #self.queue.appendleft(influenced)

    def _get_influencers_url(self, artist_id):
        return self.API_URL + 'influencers?apikey=%s&sig=%s&nameid=%s' % (self.KEY, self._get_sig(), artist_id)

    def _get_followers_url(self, artist_id):
        return self.API_URL + 'followers?apikey=%s&sig=%s&nameid=%s' % (self.KEY, self._get_sig(), artist_id)

    def _get_sig(self):
        '''
        Return a signature to encode into the GET request URL to the Rovicorp API.
        '''
        now_timestamp = int(time.time())
        if self.sig is None or now_timestamp - self.last_timestamp > self.REFRESH_PERIOD:
            # Refresh the signature.
            m = hashlib.md5()
            m.update(self.KEY)
            m.update(self.SECRET)
            m.update(str(now_timestamp))
            self.last_timestamp = now_timestamp
            self.sig = m.hexdigest() 
        return self.sig

def main():
    scraper = Scraper()

    seed_id = 'MN0000423829'
    seed_name = 'Miles Davis'

    # Load the graph so far.
    scraper.artist_ids = pickle.load(open('../data/artist_ids.pickle', 'rb'))
    scraper.influencers = pickle.load(open('../data/influencers_graph.pickle', 'rb'))
    scraper.visited = pickle.load(open('../data/visited_artists.pickle', 'rb'))
    scraper.queue = pickle.load(open('../data/bfs_queue.pickle'))

    scraper.print_stats()

    # Ignore QPS exceeded error and dump to pickle anyway.
    try:
        scraper.generate_influence_graph(seed_id, seed_name)
    except MaxQpsExceededError:
        print 'Got MaxQpsExceededError. Aborting search...'
    except:
        print 'Got Exception. Aborting search...'

    # print 'Influencers', scraper.influencers
    # print 'IDs', scraper.artist_ids
    scraper.print_stats()

    print 'Dumping to pickle files...'
    pickle.dump(scraper.artist_ids, open('artist_ids.pickle', 'wb'))
    pickle.dump(scraper.influencers, open('influencers_graph.pickle', 'wb'))
    pickle.dump(scraper.visited, open('visited_artists.pickle', 'wb'))
    pickle.dump(scraper.queue, open('bfs_queue.pickle', 'wb'))

if __name__ == '__main__':
    main()
