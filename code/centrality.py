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

closeness_set = set([864460,
1612560,
1933704,
4117251,
4999974,
6228215,
6850113,
7313165,
7570573,
8043024])

next(edges)
for line in edges:
  token = line.split(';')
  id_1 = get_id(token[0]) #turn into integer for graph node id
  id_2 = get_id(token[1])
  if id_1 in closeness_set:
    print '%d is %s' %(id_1, token[0])
  if id_2 in closeness_set:
    print '%d is %s' %(id_2, token[1])
  if id_1 not in node_set:
    undirected_graph.AddNode(id_1)
    directed_graph.AddNode(id_1)
    node_set.add(id_1)
  if id_2 not in node_set:
    undirected_graph.AddNode(id_2)
    directed_graph.AddNode(id_2)
    node_set.add(id_2)
  undirected_graph.AddEdge(id_1, id_2)
  directed_graph.AddEdge(id_1, id_2)
  
OutDegV = TIntPrV()
GetNodeOutDegV(directed_graph, OutDegV)
out_degree_dict = {}
for item in OutDegV:
  out_degree_dict[item.GetVal1()] = item.GetVal2()
sorted_degrees = sorted(out_degree_dict.items(), key=lambda x:(-x[1], x[0]))
print 'Top 10 Artists With Highest Out Degree:'
for i in range(10):
  print '%d) %s'%(i+1, label_dict[sorted_degrees[i][0]])

#per genre
#per time period
closeness_dict = {}
for node in undirected_graph.Nodes():
  closeness_dict[node.GetId()] = GetClosenessCentr(undirected_graph, node.GetId())
sorted_closeness = sorted(closeness_dict.items(), key=lambda x:(-x[1], x[0]))
print 'Top 10 Artists with Highest Closeness:'
for i in range(10):
  print '%d) %s'%(i+1, sorted_closeness[i][0])

centrality_nodes = TIntFltH()
centrality_edges = TIntPrFltH()
GetBetweennessCentr(undirected_graph, centrality_nodes, centrality_edges, 1)
betweenness_dict = {}
for item in centrality_nodes:
  betweenness_dict[item] = centrality_nodes[item]
sorted_betweenness = sorted(betweenness_dict.items(), key=lambda x:(-x[1], x[0]))
print 'Top 10 Artists with Highest Betweenness:'
for i in range(10):
  print '%d) %s'%(i+1, label_dict[sorted_betweenness[i][0]])










