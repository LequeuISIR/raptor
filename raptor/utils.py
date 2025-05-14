import logging
import re
from typing import Dict, List, Set, Optional

import numpy as np
import tiktoken
from scipy import spatial
import statistics
from sklearn.metrics import pairwise_distances

from .tree_structures import Node

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def reverse_mapping(layer_to_nodes: Dict[int, List[Node]]) -> Dict[Node, int]:
    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        for node in nodes:
            node_to_layer[node.index] = layer
    return node_to_layer


def split_text(
    text: str, tokenizer: tiktoken.get_encoding("cl100k_base"), max_tokens: int, overlap: int = 0, make: bool = False
):
    """
    Splits the input text into smaller chunks based on the tokenizer and maximum allowed tokens.
    
    Args:
        text (str): The text to be split.
        tokenizer (CustomTokenizer): The tokenizer to be used for splitting the text.
        max_tokens (int): The maximum allowed tokens.
        overlap (int, optional): The number of overlapping tokens between chunks. Defaults to 0.
    
    Returns:
        List[str]: A list of text chunks.
    """

    if make:
        delimiters = ["\n"]
        regex_pattern = "|".join(map(re.escape, delimiters))
        sentences = re.split(regex_pattern, text)
        stripped_sentences = [sent.strip() for sent in sentences if sent.strip() != ""]
        # if overlap :
        # stripped_sentences = [stripped_sentences[i-1] + stripped_sentences[i] for i in range(1, len(stripped_sentences)) ]
        return stripped_sentences
    

    # Split the text into sentences using multiple delimiters
    delimiters = [".", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)
    
    
    # Calculate the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence, token_count in zip(sentences, n_tokens):
        # If the sentence is empty or consists only of whitespace, skip it
        if not sentence.strip():
            continue
        
        # If the sentence is too long, split it into smaller parts
        if token_count > max_tokens:
            sub_sentences = re.split(r"[,;:]", sentence)
            
            # there is no need to keep empty os only-spaced strings
            # since spaces will be inserted in the beginning of the full string
            # and in between the string in the sub_chuk list
            filtered_sub_sentences = [sub.strip() for sub in sub_sentences if sub.strip() != ""]
            sub_token_counts = [len(tokenizer.encode(" " + sub_sentence)) for sub_sentence in filtered_sub_sentences]
            
            sub_chunk = []
            sub_length = 0
            
            for sub_sentence, sub_token_count in zip(filtered_sub_sentences, sub_token_counts):
                if sub_length + sub_token_count > max_tokens:
                    
                    # if the phrase does not have sub_sentences, it would create an empty chunk
                    # this big phrase would be added anyways in the next chunk append
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                        sub_length = sum(sub_token_counts[max(0, len(sub_chunk) - overlap):len(sub_chunk)])
                
                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count
            
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
        
        # If adding the sentence to the current chunk exceeds the max tokens, start a new chunk
        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(n_tokens[max(0, len(current_chunk) - overlap):len(current_chunk)])
            current_chunk.append(sentence)
            current_length += token_count
        
        # Otherwise, add the sentence to the current chunk
        else:
            current_chunk.append(sentence)
            current_length += token_count
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[float]:
    """
    Calculates the distances between a query embedding and a list of embeddings.

    Args:
        query_embedding (List[float]): The query embedding.
        embeddings (List[List[float]]): A list of embeddings to compare against the query embedding.
        distance_metric (str, optional): The distance metric to use for calculation. Defaults to 'cosine'.

    Returns:
        List[float]: The calculated distances between the query embedding and the list of embeddings.
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances


def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """
    Converts a dictionary of node indices to a sorted list of nodes.

    Args:
        node_dict (Dict[int, Node]): Dictionary of node indices to nodes.

    Returns:
        List[Node]: Sorted list of nodes.
    """
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list


def get_embeddings(node_list: List[Node], embedding_model: str) -> List:
    """
    Extracts the embeddings of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.
        embedding_model (str): The name of the embedding model to be used.

    Returns:
        List: List of node embeddings.
    """
    return [node.embeddings[embedding_model] for node in node_list]


def get_children(node_list: List[Node]) -> List[Set[int]]:
    """
    Extracts the children of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        List[Set[int]]: List of sets of node children indices.
    """
    return [node.children for node in node_list]


def get_text(node_list: List[Node]) -> str:
    """
    Generates a single text string by concatenating the text from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        str: Concatenated text.
    """
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text


def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    """
    Returns the indices of nearest neighbors sorted in ascending order of distance.

    Args:
        distances (List[float]): A list of distances between embeddings.

    Returns:
        np.ndarray: An array of indices sorted by ascending distance.
    """
    return np.argsort(distances)


# def context_aware_distance(distance_matrix: np.ndarray, all_tree_nodes: Dict[int,Node], l: float = 0.5, aggregate: str = "mean") :
#     context_aware_distance_matrix = distance_matrix.copy()
#     for row in range(len(distance_matrix)) :
#         for column in range(len(distance_matrix[0])) :
#             context_nodes_id = all_tree_nodes[column].context_nodes # - {row}
#             if aggregate == "mean" :
#                 sum = 0
#                 if len(context_nodes_id) != 0 :
#                     for node_id in context_nodes_id :
#                         sum += distance_matrix[row, node_id]
#                     mean = sum/len(context_nodes_id)
#                 else :
#                     mean = 0
#             context_aware_distance_matrix[row, column] = l*context_aware_distance_matrix[row, column] + (1-l)*mean 
#     return context_aware_distance_matrix

def context_aware_distance(distance_matrix: np.ndarray, all_tree_nodes: Dict[int,Node], l: float = 0.3, aggregate: str = "mean") :

    context_aware_distance_matrix = distance_matrix.copy()
    for row in range(len(distance_matrix)) :
        for column in range(len(distance_matrix[0])) :
            column_context_nodes_id = all_tree_nodes[column].context_nodes # - {row}
            row_context_nodes_id = all_tree_nodes[column].context_nodes # - {row}
            if aggregate == "mean" :
                sum = 0
                if len(column_context_nodes_id) != 0 :
                    for column_context_node_id in column_context_nodes_id :
                        sum += distance_matrix[row, column_context_node_id]
                    
                if len(row_context_nodes_id) != 0 :
                        for row_context_node_id in row_context_nodes_id :
                                sum += distance_matrix[column, row_context_node_id]

                if sum != 0 : # it means one of the context nodes list is not empty
                    mean = sum/(len(column_context_nodes_id) + len(row_context_nodes_id))
                else :
                    mean = 0
            context_aware_distance_matrix[row, column] = l*context_aware_distance_matrix[row, column] + (1-l)*mean

    return context_aware_distance_matrix

# def context_aware_distance(node_1: Node, 
#                            node_2: Node, 
#                            distance_metric: str = "cosine", 
#                            l: float = 0.5,
#                            aggregate: str = "mean",
#                            all_tree_node: Optional[Dict[int, Node]] = None) -> float:
#     """
#     Returns the distance from node_1 to node_2. It is not the same as the distance from node_2 to node_1.
#     it calculates the distance of node_1.text to node_2.text, and the avg/(max?) of the distance of node_1.text to the context of node_2 .
#     l is lambda: if l=1, only take into account the sentence. if l=0, only take into account the context.
#     """
#     distance_metrics = {
#         "cosine": spatial.distance.cosine,
#         "L1": spatial.distance.cityblock,
#         "L2": spatial.distance.euclidean,
#         "Linf": spatial.distance.chebyshev,
#     }

#     if distance_metric not in distance_metrics:
#         raise ValueError(
#             f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
#         )

#     # Distance between node_1 and node_2 texts
#     text_distance = distance_metrics[distance_metric](node_1.embeddings["EMB"], node_2.embeddings["EMB"])
    
#     # Distance between node_1 atext and node_2 context
#     embeddings =  [all_tree_node[id].embeddings["EMB"] for id in node_2.context_nodes]
#     distances = [
#         distance_metrics[distance_metric](node_1.embeddings["EMB"], embedding)
#         for embedding in embeddings
#     ]

#     if aggregate == "mean" :
#         context_distance = statistics.mean(distances)
    
#     return l*text_distance + (1-l)*context_distance
