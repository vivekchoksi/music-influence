from snap import *

edges = open("../data/edges.csv")
labels = open("../data/node_labels.csv")

undirected_graph = TUNGraph.New()
directed_graph = TNGraph.New()
node_set = set()
label_dict = {}
second_label_dict = {}

def get_id(identifier):
  return abs(hash(identifier)) % (10**8)

next(labels)
for line in labels:
  token = line.split(';')
  node_id = get_id(token[0])
  label_dict[node_id] = token[1]
labels.close()

next(edges)
for line in edges:
  token = line.split(';')
  id_1 = get_id(token[0]) #turn into integer for graph node id
  id_2 = get_id(token[1])
  if id_1 not in node_set:
    undirected_graph.AddNode(id_1)
    directed_graph.AddNode(id_1)
    node_set.add(id_1)
  if id_2 not in node_set:
    undirected_graph.AddNode(id_2)
    directed_graph.AddNode(id_2)
    node_set.add(id_2)
  undirected_graph.AddEdge(id_1, id_2)
  directed_graph.AddEdge(id_2, id_1) #reverse for pagerank

PRankH = TIntFltH()
GetPageRank(directed_graph, PRankH)
pagerank_dict = {}
for item in PRankH:
  pagerank_dict[item] = PRankH[item]
sorted_pagerank = sorted(pagerank_dict.items(), key=lambda x:(-x[1], x[0]))
print 'Top 10 Artists with Highest Pagerank:'
for i in range(10):
  print '%d) %s'%(i+1, label_dict[sorted_pagerank[i][0]])

