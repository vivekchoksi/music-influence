import ast

def main():
  artist_songs = open('../data/artist_to_songs.csv', 'r')
  song_file = open('../data/evolution.csv', 'r')
  next(song_file)
  vector_dictionary = {}
  line_count = 1
  for line in song_file:
    line_count += 1
    tokens = line.split(',')
    topic_vector = tokens[11:27]
    vector_dictionary[line_count] = topic_vector
  song_file.close()

  total_songs = 0
  next(artist_songs)
  artist_to_songs = {}
  for line in artist_songs:
    tokens = line.split(';')
    artist_to_songs[tokens[0].strip()] = ast.literal_eval(tokens[1].strip())
    total_songs += len(ast.literal_eval(tokens[1]))
  print 'total songs: %d' %(total_songs)

  write_dictionary = {}

  for artist in artist_to_songs:
    song_indices = artist_to_songs[artist]
    write_dictionary[artist] = []
    for index in song_indices:
      write_dictionary[artist].append(vector_dictionary[index])

  with open('../data/song_vectors.csv', 'w') as write_file:
    for artist_id in write_dictionary:
      write_file.write(str(artist_id) + ';' + str(write_dictionary[artist_id]) + '\n')

  artist_songs.close()
  

if __name__ == "__main__":
  main()