import pdb
import cPickle as pickle
import re

ARTISTS_TO_SONGS_PICKLE = 'data/artist_to_songs.pickle'
EVOLUTION_FILE = 'data/evolution.csv'
ARTIST_TO_YEARS_OUT_PICKLE = 'data/artists_to_years.pickle'

def get_artists_to_years():
    artists_to_years = {}
    artists_to_songs = pickle.load(open(ARTISTS_TO_SONGS_PICKLE, 'rb'))
    evolution = open(EVOLUTION_FILE, 'rb')
    songs = evolution.readlines()


    for artist in artists_to_songs:
        artists_to_years[artist] = set()
        for song in artists_to_songs[artist]:
            song_data = songs[song - 1].split(',')
            year = re.split('-| ', song_data[6])[0]

            try:
                year = int(year)
            except:
                print 'Malformed year:', year
                for datum in song_data:
                    if len(datum) == 4 and is_int(datum) and int(datum) < 2100 and int(datum) > 1900:
                        year = int(datum)
                        print '\tResolving year to:', year
                        break
            artists_to_years[artist].add(year)

    return artists_to_years

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def dump(artists_to_years, dump_filename):
    print 'Dumping to file:', dump_filename, '...'
    pickle.dump(artists_to_years, open(dump_filename, 'wb'))
    print 'Done.'

if __name__ == '__main__':
    artists_to_years = get_artists_to_years()
    dump(artists_to_years, ARTIST_TO_YEARS_OUT_PICKLE)


