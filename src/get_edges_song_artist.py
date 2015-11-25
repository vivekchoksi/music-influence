#We just want a graph with artists of the song set we have (evolution.csv)

import datetime
import allmusic_scraper as sc
import generate_edgelist as ge
import pickle

def is_date(string):
  try:
    datetime.datetime.strptime(string, '%Y-%m-%d')
    return True
    return True
  except ValueError:
    return False

def find_year_index(tokens):
  for i in range(len(tokens)):
    if is_date(tokens[i]):
      return i

def is_same_artist(artist_one, artist_two):
  if artist_one.lower() == artist_two.lower():
    return True
  if 'featuring' in artist_one.lower():
    tokens = artist_one.lower().split("featuring")
    print tokens
    print tokens[0].strip()
    print tokens[1].strip()
    print artist_two.lower()
    if tokens[0].strip() == artist_two.lower() or tokens[1].strip() == artist_two.lower():
      return True
  return False
  # can add to this later -- might not be exactly equal (i.e.Beyonce with accented 'e' vs no accent)
  # - use some form of regex or check if half the words are the same?

def find_song(song_title, artist_name, scraper):
  song_json_response = scraper.get_song(song_title) #this only gets one song -- problem: multiple with same name
  artist_list = song_json_response['song']['primaryArtists']
  is_same = False
  for artist_retrieved in artist_list: #in the case of 'feat' or collab
    print artist_retrieved
    if is_same_artist(artist_name, artist_retrieved['name']): 
      artist_id = artist_retrieved['id']
      is_same = True
      break
  if is_same:
    return True, artist_id
  else:
    return False, None

def find_artist(artist_name, scraper, has_influencers_dict, has_followers_dict):
  artist_json_response = scraper.get_artist(artist_name)
  if artist_json_response != -1:
    artist_retrieved = artist_json_response['name']['name']
    if is_same_artist(artist_name, artist_retrieved): #make this check, because sometimes we get back rubbish
      artist_id = artist_json_response['name']['ids']['nameId']
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

def create_data_files(artist_ids, scraper, has_influencers_dict, has_followers_dict):
  for artist_id in artist_ids:
    if not has_influencers_dict[artist_id]:
      print '%s has no influencers' % artist_id
    else:
      scraper._get_influencers(artist_id, artist_ids)
    if not has_followers_dict[artist_id]:
      print '%s has no followers' % artist_id
    else:
      scraper._get_followers(artist_id, artist_ids)
    
  # get influencers for each of the artists, make sure that's in the artist set as well


def main():
  song_file = open('../data/evolution.csv', 'r')
  for i in range(1001): 
    next(song_file)
  song_to_artist = {}
  songs = []
  num_lines = 0
  for line in song_file:
    if num_lines == 200:
      break
    num_lines += 1
    line = line.replace('"', '').strip()
    tokens = line.split(',')
    year_index = find_year_index(tokens)
    artist_name = tokens[1]
    song_title = tokens[year_index-1]
    songs.append((song_title, tokens[year_index]))
    song_to_artist[(song_title, tokens[year_index])] = artist_name
    # THIS ASSUMES THAT EVERY SONG HAS A DATE AND AN ARTIST NAME, AND SONG TITLE IS ALWAYS BEFORE DATE
  scraper = sc.Scraper()

  # looking at songs, since artist name could be a combination of diff. artists (i.e. feat)
  artist_ids = set()
  artist_dict = {}
  has_influencers_dict = {}
  has_followers_dict = {}
  for song in songs: 
    artist_name = song_to_artist[song]
    is_found, artist_id = find_artist(artist_name, scraper, has_influencers_dict, has_followers_dict)
    if not is_found:
      print 'Artist was not found'
      #is_found, artist_id = find_song(song, artist_name, scraper, has_followers_dict)
      #if artist_id is None:
        #print 'Could not find matching song/artist combo'
    print artist_name, song, artist_id
    if artist_id != None:
      del song_to_artist[song]
      artist_ids.add(artist_id)
      artist_dict[artist_id] = artist_name
    
  print 'We were not able to find the following song/artists in allmusic:'
  for key in song_to_artist:
    print key, song_to_artist[key]

  create_data_files(artist_ids, scraper, has_influencers_dict, has_followers_dict)
  pickle.dump(scraper.influencers, open('../data/influencers_song_artists_only_graph.pickle', 'wb'))
  pickle.dump(artist_dict, open('../data/song_artists_only.pickle', 'wb'))
  ge.generate_edge_lists('../data/influencers_song_artists_only_graph.pickle', 'song_artists_only_edges')
  ge.generate_node_labels('../data/song_artists_only.pickle', 'song_artists_only_labels')
if __name__ == "__main__":
  main()
  




