#We just want a graph with artists of the song set we have (evolution.csv)

import datetime
import allmusic_scraper as sc
import generate_edgelist as ge
import pickle
import os.path
import copy

genres = {}
time_active = {}

def is_date(string):
  try:
    datetime.datetime.strptime(string, '%Y-%m-%d')
    return True
  except ValueError:
    return False

def find_year_index(tokens):
  for i in range(len(tokens)):
    if is_date(tokens[i]):
      return i

# expect artist_one to be from evolution.csv and artist_two to be from allmusic.com
def is_same_artist(artist_one, artist_two):
  if artist_one.lower() == artist_two.lower():
    return True
  if 'featuring' in artist_one.lower():
    tokens = artist_one.lower().split("featuring")
    if tokens[0].strip() == artist_two.lower() or tokens[1].strip() == artist_two.lower():
      return True
  return False
  # can add to this later -- might not be exactly equal (i.e.Beyonce with accented 'e' vs no accent)
  # - use some form of regex or check if half the words are the same?

def add_genre(artist_id, json_response):
  if json_response is not None:
    for genre in json_response:
      if artist_id in genres:
        genres[artist_id].add(genre['name'])
      else:
        genres[artist_id] = set([genre['name']])

def add_time_active(artist_id, json_response):
  if json_response is not None:
    time_active[artist_id] = json_response

def find_artist(artist_name, scraper, has_influencers_dict, has_followers_dict):
  artist_json_response = scraper.get_artist(artist_name)
  if artist_json_response != -1:
    artist_retrieved = artist_json_response['name']['name']
    if is_same_artist(copy.deepcopy(artist_name), artist_retrieved): #make this check, because sometimes we get back rubbish
      artist_id = artist_json_response['name']['ids']['nameId']
      ### populate genre and active pickle here ###
      add_genre(artist_id, artist_json_response['name']['musicGenres'])
      add_time_active(artist_id, artist_json_response['name']['active'])
      if artist_json_response['name']['influencersUri'] != None:
        has_influencers_dict[artist_id] = True
      else:
        has_influencers_dict[artist_id] = False
      if artist_json_response['name']['followersUri'] != None:
        has_followers_dict[artist_id] = True
      else:
        has_followers_dict[artist_id] = False
      return True, artist_id
    else:
      return False, None
  else:
    return False, None

def create_data_files(artist_dict, scraper, has_influencers_dict, has_followers_dict, influencers_dict):
  print 'Creating edges...'
  for a in artist_dict:
    if a in has_influencers_dict and has_influencers_dict[a] == True:
      scraper._get_influencers(a)
    if a in has_followers_dict and has_followers_dict[a] == True:
      scraper._get_followers(a)

def get_artist_and_song(line):
  line = line.replace('"', '').strip()
  tokens = line.split(',')
  year_index = find_year_index(tokens) # use this for orientation (each song might be missing diff. attributes)
  artist_name = tokens[1]
  song_title = tokens[year_index-1]
  return artist_name, song_title

def add_song(artist_to_songs, artist_name, song_index):
  if artist_name in artist_to_songs:
    artist_to_songs[artist_name].append(song_index)
  else:
    artist_to_songs[artist_name] = [song_index]

def get_artist_to_songs_dict():
  file_open = open('../data/artist_to_songs.csv', 'r')
  to_return = {}
  next(file_open)
  for line in file_open:
    tokens = line.split(';')
    to_return[tokens[0].strip()] = tokens[1].strip()
  return to_return
  #if os.path.isfile('../data/artist_to_songs.pickle'):
  #  return pickle.load(open('../data/artist_to_songs.pickle', 'r'))
  #else:
  #  return {}

def get_artist_labels():
  if os.path.isfile('../data/song_artists_labels_only.pickle'):
    return pickle.load(open('../data/song_artists_labels_only.pickle', 'r'))
  else:
    return {}

def load_genres():
  if os.path.isfile('../data/genres.pickle'):
    genres = pickle.load(open('../data/genres.pickle', 'r'))

def load_time_active():
  if os.path.isfile('../data/time_active.pickle'):
    time_active = pickle.load(open('../data/time_active.pickle','r'))

def get_seen_influencers():
  if os.path.isfile('../data/seen_influencers.pickle'):
    return pickle.load(open('../data/seen_influencers.pickle', 'r'))
  else:
    return {}

def generate_influencers_graph(to_populate, all_influencers_pickle, artist_ids):
  all_influencers = pickle.load(open(all_influencers_pickle, 'r'))
  subgraph_influencers = {}
  for person in all_influencers:
    if person not in artist_ids: continue
    subgraph_influencers[person] = set()
    for influencer_id in all_influencers[person]:
      if influencer_id in artist_ids:
        subgraph_influencers[person].add(influencer_id)
  pickle.dump(subgraph_influencers, open(to_populate, 'wb'))

def main():
  load_genres()
  load_time_active()
  song_file = open('../data/evolution.csv', 'r')
  num_already_looked_at = 2010+1 #how many songs i've already looked at (first line is just column titles)
  for i in range(num_already_looked_at): 
    next(song_file)
  artist_to_songs = get_artist_to_songs_dict() #read from pickle file (if exists)
  song_to_artist = {}
  num_lines = 0
  for line in song_file:
    if num_lines == 6: #how many we want to scrape this time (for query limit)
      break
    num_lines += 1
    artist_name, song_title = get_artist_and_song(line)
    add_song(artist_to_songs, artist_name, num_already_looked_at+num_lines)
    song_to_artist[song_title] = artist_name
    # THIS ASSUMES THAT EVERY SONG HAS A DATE AND AN ARTIST NAME, AND SONG TITLE IS ALWAYS BEFORE DATE
  scraper = sc.Scraper()
  artist_ids = set()
  artist_dict = {} # for node labels
  has_influencers_dict = {}
  has_followers_dict = {}
  for song in song_to_artist: 
    artist_name = song_to_artist[song]
    is_found, artist_id = find_artist(copy.deepcopy(artist_name), scraper, has_influencers_dict, has_followers_dict)
    if artist_id in artist_ids: continue
    if artist_id != None:
      artist_to_songs[artist_id] = artist_to_songs[artist_name]
    artist_to_songs.pop(artist_name, None)
    if not is_found:
      print 'Artist was not found: %s' %artist_name
    else:
      print artist_name, song, artist_id
    if artist_id != None:
      artist_ids.add(artist_id)
      artist_dict[artist_id] = artist_name
  print 'FINISHED'
  labels_already_generated = get_artist_labels()
  # add all the artists we've already seen, so we include them in our regenerated influence graph
  for artist_id in labels_already_generated:
    artist_ids.add(artist_id)
  influencers_dict = get_seen_influencers()
  create_data_files(artist_dict, scraper, has_influencers_dict, has_followers_dict, influencers_dict)
  for artist_id in labels_already_generated:
    artist_dict[artist_id] = labels_already_generated[artist_id]
  print 'Generated labels'
  influencers_dict.update(scraper.influencers)
  
  pickle.dump(artist_dict, open('../data/song_artists_labels_only.pickle', 'wb'))
  pickle.dump(artist_to_songs, open('../data/artist_to_songs.pickle', 'wb'))
  pickle.dump(genres, open('../data/genres.pickle', 'wb'))
  pickle.dump(time_active, open('../data/time_active.pickle', 'wb'))
  pickle.dump(influencers_dict, open('../data/seen_influencers.pickle', 'wb'))


  #to avoid getting everyone's influencers every single time
  generate_influencers_graph('../data/song_artists_edges_only.pickle', '../data/seen_influencers.pickle', artist_ids)
  
  # generate csv files
  ge.generate_edge_lists('../data/song_artists_edges_only.pickle', 'song_artists_only_edges')
  ge.generate_node_labels('../data/song_artists_labels_only.pickle', 'song_artists_only_labels')
  ge.generate_artist_songs('../data/artist_to_songs.pickle', 'artist_to_songs')
  ge.generate_genres('../data/genres.pickle', 'genres')
  ge.generate_time_active('../data/time_active.pickle', 'time_active')

if __name__ == "__main__":
  main()
  




