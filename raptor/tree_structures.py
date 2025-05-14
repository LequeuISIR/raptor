from typing import Dict, List, Set, Optional


class Node:
    """
    Represents a node in the hierarchical tree structure.
    """

    def __init__(self, text: str, index: int, children: Set[int], embeddings, speaker: str, context_nodes: Optional[Set[int]] = None, weight: float = 0.0) -> None:
        self.speaker = speaker
        self.text = text
        self.index = index
        self.children = children
        self.embeddings = embeddings
        self.context_nodes = context_nodes
        self.weight = weight



class Tree:
    """
    Represents the entire hierarchical tree structure.
    """

    def __init__(
        self, all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes

    def compute_nodes_weights(self) :

        leaf_number = len(self.leaf_nodes)
        print("number of leafs:", leaf_number)
        for id, node in self.all_nodes.items() :
            num_children = self.get_number_of_leaf_children(node)
            node.weight = num_children / leaf_number
            print("node id:", id, "weight:", node.weight)
            self.all_nodes[id] = node
    
    def get_number_of_leaf_children(self, node) :
        if type(node) == int:
            node = self.all_nodes[node]

        if len(node.children) == 0 : #its a leaf node
            return 1
        sum = 0
        for child in node.children:
            sum += self.get_number_of_leaf_children(self.all_nodes[child])
        return sum
        
