#!/usr/bin/python

# CS224w autumn 2015
# See http://prod-doc.rovicorp.com/mashery/index.php/Authentication-Code-Examples


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
    api_url = 'http://api.rovicorp.com/data/v1.1/name/'

    key = secrets.API_KEY
    secret = secrets.SHARED_SECRET

    # The interval, in seconds, after which the signature should be recalculated in self.get_sig().
    refresh_period = 100

    # The amount of time, in seconds, to wait after issuing each API request. This rate limit is imposed to avoid
    # exceeding the maximum allowable QPS.
    wait_time = 0.2

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

    def __init__(self):
        self.last_timestamp = int(time.time())

    def get_influencers(self, artist, queue):
        # Get influencers.
        url = self.get_influencers_url(artist)
        response = self.get_response(url)

        if response == -1:
            # Error.
            return
        else:
            influencers_response = response['influencers']

        print self.artist_ids[artist], 'has', len(influencers_response), 'influencers'
        for influencer in influencers_response:
            # Map influencer id to name.
            if influencer['id'] not in self.artist_ids:
                self.artist_ids[influencer['id']] = influencer['name']

            self.create_influence_edge(influencer['id'], artist, queue)

    def get_followers(self, artist, queue):
        # Get followers.
        url = self.get_followers_url(artist)
        response = self.get_response(url)

        if response == -1:
            # Error.
            return
        else:
            followers_response = response['followers']

        print self.artist_ids[artist], 'has', len(followers_response), 'followers'

        for follower in followers_response:
            # Map follower id to name.
            if follower['id'] not in self.artist_ids:
                self.artist_ids[follower['id']] = follower['name']

            self.create_influence_edge(artist, follower['id'], queue)

    def get_response(self, url):
        '''
        Return the contents of a GET request to the specified url; return -1 on error.
        '''
        resp = requests.get(url)
        # print 'GET', url

        # Rate limit to avoid exceeding max allowable QPS.
        time.sleep(self.wait_time)

        if resp.status_code != 200:
            print 'Non-200 response:', resp.content

            if resp.content.find('Developer Over Rate') != -1:
                raise MaxQpsExceededError
            elif resp.content.find('Not Found') != -1:
                print 'No data received from request', url

            return -1

        # Get json list of followers.
        return resp.json()

    def create_influence_edge(self, influencer, influenced, queue):
        # Add influencer to graph.
        if influenced not in self.influencers:
            self.influencers[influenced] = set()
        self.influencers[influenced].add(influencer)

        if influencer not in self.visited:
            queue.appendleft(influencer)
        if influenced not in self.visited:
            queue.appendleft(influenced)

    def calculate_influencers(self, seed_id, seed_name, max_artists=None):
        '''
        Given a seed artist, perform breadth-first search on the influence graph.
        '''

        queue = deque([seed_id])
        self.artist_ids[seed_id] = seed_name
     
        while len(queue) > 0:
            artist = queue.pop()
            if artist not in self.visited:
                self.visited.add(artist)
                self.get_influencers(artist, queue)
                self.get_followers(artist, queue)

                if max_artists is not None and len(self.visited) > max_artists:
                    print 'Thelonious Monk self.visited:', u'MN0000490416' in self.visited
                    print 'Ella Fitzerald self.visited:', u'MN0000184502' in self.visited
                    break


    def get_influencers_url(self, artist_id):
        return self.api_url + 'influencers?apikey=%s&sig=%s&nameid=%s' % (self.key, self.get_sig(), artist_id)

    def get_followers_url(self, artist_id):
        return self.api_url + 'followers?apikey=%s&sig=%s&nameid=%s' % (self.key, self.get_sig(), artist_id)

    def get_sig(self):
        now_timestamp = int(time.time())
        if self.sig is None or now_timestamp - self.last_timestamp > self.refresh_period:
            m = hashlib.md5()
            m.update(self.key)
            m.update(self.secret)
            m.update(str(now_timestamp))
            self.last_timestamp = now_timestamp
            self.sig = m.hexdigest()
            return self.sig
        else:
            return self.sig


def main():
    seed_id = 'MN0000423829'
    seed_name = 'Miles Davis'
    scraper = Scraper()

    # Ignore exceptions and dump to pickle anyway.
    try:
        scraper.calculate_influencers(seed_id, seed_name, 1000)
    except MaxQpsExceededError:
        print 'Got MaxQpsExceededError. Aborting search...'

    # print 'Influencers', scraper.influencers
    # print 'IDs', scraper.artist_ids

    print 'Number of artists visited:', len(scraper.visited)
    print 'Number of nodes in graph:', len(scraper.artist_ids)
    print 'Dumping to pickle files...'
    pickle.dump(scraper.influencers, open('influencers_graph.pickle', 'wb'))
    pickle.dump(scraper.artist_ids, open('artist_ids.pickle', 'wb'))
    pickle.dump(scraper.visited, open('visited_artists.pickle', 'wb'))

# Commands to load pickle dumps.
# influencers = pickle.load(open('influencers_graph.pickle', 'rb'))
# artist_ids = pickle.load(open('artist_ids.pickle', 'rb'))

if __name__ == '__main__':
    main()