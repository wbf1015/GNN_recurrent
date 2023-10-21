import numpy as np

def check_randancy(path="../data/cora/", dataset="cora"):
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    unique_edges = set()
    for edge in edges_unordered:
        sorted_edge = tuple(sorted(edge))
        unique_edges.add(sorted_edge)
    
    if len(unique_edges) == len(edges_unordered):
        print("没有冗余的边")
    else:
        print("存在冗余的边")
        print("冗余的边数为", len(edges_unordered)-len(unique_edges))

check_randancy()