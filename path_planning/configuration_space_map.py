from typing import List, Callable, Optional
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# EUCLEDIAN_DISTANCE_FUNCTION is a lambda function that calculates
# the eucledian distance between two points in n-dimensional space
# the two inputs must be np.ndarrays of the same size
EUCLEDIAN_DISTANCE_FUNCTION = lambda x,y:np.linalg.norm(x-y)

# set up logger
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
#log to console
log.addHandler(logging.StreamHandler())


class CollisionMap:
    """
    Graph representation of the configuration collision free space
    """
    def __init__(self, nodes:np.ndarray, edge_indices:np.ndarray, edge_weights:np.ndarray):
        self.nodes = nodes
        self.edge_indices = edge_indices
        self.edge_weights = edge_weights
    
    def nearest_neighbor(
            self, 
            node:np.ndarray,
            distance_func:Callable[[np.ndarray,np.ndarray],float]=EUCLEDIAN_DISTANCE_FUNCTION)->int:
        """
        Find the nearest node to the given node
        args:
            node: np.ndarray of size (n,) (must match self.nodes of size (m,n))
                the node to find the nearest neighbor to
            distance_func: Callable[[np.ndarray,np.ndarray],float]
                the distance function to use
        """
        min_dist = np.inf
        nearest_node = -1
        for i in range(len(self.nodes)):
            dist = distance_func(node,self.nodes[i])
            if dist < min_dist:
                min_dist = dist
                nearest_node = i
        return nearest_node
    
    def nodes_in_neighborhood(
            self, 
            node:np.ndarray, 
            max_distance:float,
            distance_func:Callable[[np.ndarray,np.ndarray],float]=EUCLEDIAN_DISTANCE_FUNCTION)->List[int]:
        """ 
        Finds the nodes in the neighborhood of the given node and max distance
        args:
            node: np.ndarray of size (n,) (must match self.nodes of size (m,n))
                the node to find the neighborhood of
            max_distance: float
                the maximum distance to consider
            distance_func: Callable[[np.ndarray,np.ndarray],float]
                the distance function to use
        """
        neighborhood = []
        for i in range(len(self.nodes)):
            if distance_func(node,self.nodes[i]) <= max_distance:
                neighborhood.append(i)
        return neighborhood
    
    def a_star(
            self, 
            start_node_index:int, 
            end_node_index:int, 
            node_distance_function:Callable=EUCLEDIAN_DISTANCE_FUNCTION)->Tuple[List[int],float]:
        
        """
        Implementation of the A* algorithm for graph search
        for more information about A*: https://en.wikipedia.org/wiki/A*_search_algorithm
        args:
            start_node_index: int
                the index of the start node in self.nodes
            end_node_index: int
                the index of the end node in self.nodes
            node_distance_function: Callable[[np.ndarray,np.ndarray],float]
                the distance function to use, defaults to EUCLEDIAN_DISTANCE_FUNCTION
        returns:
            Tuple[List[int],float]
                - the path from the start node to the end node as a list of node indices (empty list if no path is found)
                - the cost of the path
        """
        # list of nodes that are open for evaluation
        open_list = [start_node_index]
        # list of nodes that have been evaluated
        closed_list = []
        # g is the cost to reach the node from the start node
        g = {i:np.inf for i in range(len(self.nodes))}
        g[start_node_index] = 0
        # f is the sum of g and h where h is the heuristic (estimated cost to reach the end node from the current node)
        f = {i:np.inf for i in range(len(self.nodes))}
        f[start_node_index] = node_distance_function(self.nodes[start_node_index],self.nodes[end_node_index])
        # parent dictionary to reconstruct the path
        parent = {}
        while open_list:
            current_node = open_list[0]
            if current_node == end_node_index:
                # we reached the end node 
                # reconstruct the path from the parent dictionary
                path = []
                while current_node != -1:
                    path.append(current_node)
                    current_node = parent.get(current_node,-1)
                path.reverse()
                return path,g[end_node_index]
            
            open_list.remove(current_node)
            closed_list.append(current_node)
            for k,edge_index in enumerate(self.edge_indices):
                if current_node in edge_index:
                    neighbor_node_index = edge_index[0] if edge_index[1] == current_node else edge_index[1]
                    if  neighbor_node_index in closed_list:
                        continue
                    tentative_g = g[current_node] + self.edge_weights[k]
                    if neighbor_node_index not in open_list:
                        open_list.append(neighbor_node_index)
                    elif tentative_g >= g[neighbor_node_index]:
                        continue
                    # if the tentative_g of taking this path is lower than the previous one, we update the lowest cost path 
                    parent[neighbor_node_index] = current_node
                    g[neighbor_node_index] = tentative_g
                    f[neighbor_node_index] = g[neighbor_node_index] + node_distance_function(self.nodes[neighbor_node_index],self.nodes[end_node_index])
                    open_list.sort(key=lambda x:f[x])
        # failed to find a path, so we return an empty list and a cost of infinity
        return [],np.inf
    
    @classmethod
    def from_rrt(cls, 
            n_samples:int,
            sampler_func:Callable[...,np.ndarray],
            step_size:float,
            node_collision_func:Callable,
            start_node:np.ndarray,
            end_node:np.ndarray,
            end_node_check_ratio:float=0.1,
            solution_atol=5,
            edge_collision_func:Optional[Callable]=None,
            distance_func:Callable[[np.ndarray,np.ndarray],float]=EUCLEDIAN_DISTANCE_FUNCTION,
            )->Tuple['CollisionMap',List[int]]:
        """
        Creates a graph and path using the RRT algorithm
        for more information about RRT: https://en.wikipedia.org/wiki/Rapidly_exploring_random_tree

        args:
            n_samples: int
                the max number of samples to generate (could terminate early if a path is found)
            sampler_func: Callable[...,np.ndarray]
                a function that generates a random sample. Shoud return a np.ndarray of size (n,)
            step_size: float
                the maximum distance to move towards the sample
            node_collision_func: Callable
                a function that checks if a node is in collision
            start_node: np.ndarray
                the start node vector of size (n,)
            end_node: np.ndarray
                the end node vector of size (n,)
            end_node_check_ratio: float
                the probability of sampling the end node directly. Should be bounded between 0 and 1
            edge_collision_func: Optional[Callable]
                a function that checks if an edge is in collision
                we can sometimes omit this function if the step size is small enough
            distance_func: Callable[[np.ndarray,np.ndarray],float]
                the distance function to use, defaults to EUCLEDIAN_DISTANCE_FUNCTION
        """
        
        start_node = np.array(start_node)
        end_node = np.array(end_node)
        output = cls(nodes= [start_node],edge_indices = [], edge_weights = [])
        parent = {}
        solution_found = False
        for i in tqdm(range(n_samples)):
            if solution_found:
                break
            if np.random.rand() < end_node_check_ratio:
                sample = end_node
                nearest_node = output.nearest_neighbor(sample,distance_func=distance_func)
            else:
                sample = np.array(sampler_func())
                nearest_node = output.nearest_neighbor(sample,distance_func=distance_func)

            dist = distance_func(output.nodes[nearest_node],sample)
            if dist > step_size:
                sample = output.nodes[nearest_node] + step_size*(sample-output.nodes[nearest_node])/np.linalg.norm(sample-output.nodes[nearest_node])
                dist = step_size

            if distance_func(sample,end_node) < solution_atol:
                if edge_collision_func is not None:
                    if edge_collision_func(output.nodes[nearest_node],sample):
                        continue
                print("Solution found")
                output.nodes.append(sample)
                parent[len(output.nodes)-1] = nearest_node
                output.edge_indices.append([nearest_node,len(output.nodes)-1])
                output.edge_weights.append(distance_func(output.nodes[nearest_node],sample))
                solution_found = True
                break
            
            if node_collision_func(sample):
                log.debug(f"collision detected at {sample}")
                continue
            if edge_collision_func is not None:
                if edge_collision_func(output.nodes[nearest_node],sample):
                    log.debug(f"edge collision detected at {sample}")
                    continue
            log.debug(f"creating node at {sample}")
            output.nodes.append(sample)
            parent[len(output.nodes)-1] = nearest_node
            output.edge_indices.append([nearest_node,len(output.nodes)-1])
            output.edge_weights.append(dist)
        path = []
        if solution_found:
            current_node = len(output.nodes)-1
            while current_node != 0:
                if current_node in path:
                    raise Exception(f"Cycle detected in path.path {path}")
                path.append(current_node)
                current_node = parent[current_node]
            path.append(0)
            path.reverse()
        output.nodes = np.array(output.nodes)
        output.edge_indices = np.array(output.edge_indices)
        output.edge_weights = np.array(output.edge_weights)
        return output,path
    
    @classmethod
    def from_rrt_star(cls,
            n_samples:int,
            sampler_func:Callable[...,np.ndarray],
            step_size:float,
            node_collision_func:Callable,
            start_node:np.ndarray,
            end_node:np.ndarray,
            connect_neighbors_radius:float,
            end_node_check_ratio:float=0.1,
            edge_collision_func:Optional[Callable]=None,
            distance_func:Callable[[np.ndarray,np.ndarray],float]=EUCLEDIAN_DISTANCE_FUNCTION,
            )->Tuple['CollisionMap',List[int]]:
        """
        Creates a graph and path using the RRT* algorithm (variant of RRT that tries to optimize the path)

        args:
            n_samples: int
                the max number of samples to generate (could terminate early if a path is found)
            sampler_func: Callable[...,np.ndarray]
                a function that generates a random sample. Shoud return a np.ndarray of size (n,)
            step_size: float
                the maximum distance to move towards the sample
            node_collision_func: Callable
                a function that checks if a node is in collision
            start_node: np.ndarray
                the start node vector of size (n,)
            end_node: np.ndarray
                the end node vector of size (n,)
            connect_neighbors_radius: float
                the maximum distance to connect neighbors
            end_node_check_ratio: float
                the probability of sampling the end node directly. Should be bounded between 0 and 1
            edge_collision_func: Optional[Callable]
                a function that checks if an edge is in collision
                we can sometimes omit this function if the step size is small enough
            distance_func: Callable[[np.ndarray,np.ndarray],float]
                the distance function to use, defaults to EUCLEDIAN_DISTANCE_FUNCTION
        returns:
            Tuple['CollisionMap',List[int]]
                - the graph representation of the map
                - the path as a list of node indices
        """
        start_node = np.array(start_node)
        end_node = np.array(end_node)
        output = cls(nodes= [start_node],edge_indices = [], edge_weights = [])
        parent = {}
        g = {0:0}
        for i in range(n_samples):
            if np.random.rand() < end_node_check_ratio:
                sample = end_node
            else:
                sample = np.array(sampler_func())
            nearest_node = output.nearest_neighbor(sample)
            if edge_collision_func is not None:
                if np.allclose(sample,end_node):
                    if not edge_collision_func(output.nodes[nearest_node],sample):
                        # we have found a solution that directly connects to the end node
                        output.nodes.append(sample)
                        parent[len(output.nodes)-1] = nearest_node
                        output.edge_indices.append([nearest_node,len(output.nodes)-1])
                        output.edge_weights.append(distance_func(output.nodes[nearest_node],sample))
                        break
            dist = distance_func(output.nodes[nearest_node],sample)
            if dist > step_size:
                sample = output.nodes[nearest_node] + step_size*(sample-output.nodes[nearest_node])/dist
                dist = step_size
            if node_collision_func(sample):
                continue
            if edge_collision_func is not None:
                if edge_collision_func(output.nodes[nearest_node],sample):
                    continue
            output.nodes.append(sample)
            sample_index = len(output.nodes)-1
            parent[sample_index] = nearest_node
            output.edge_indices.append([nearest_node,len(output.nodes)-1])
            output.edge_weights.append(dist)
            g[sample_index] = g[nearest_node] + dist

            neighborhood = output.nodes_in_neighborhood(sample,connect_neighbors_radius)

            for neighbor in neighborhood:
                if neighbor == nearest_node:
                    # we already evaluated this connection
                    continue
                if sample_index == neighbor:
                    # self connection
                    continue
                for edge_index in output.edge_indices:
                    if sample_index in edge_index and neighbor in edge_index:
                        # edge already exists
                        continue
                neighbor_dist = distance_func(output.nodes[neighbor],sample)
                cost_through_neighbor = g[neighbor] + neighbor_dist
                if cost_through_neighbor < g[sample_index]:
                    if edge_collision_func is not None:
                        if edge_collision_func(output.nodes[sample_index],output.nodes[neighbor]):
                            # edge collides with obstacle so we ignore it
                            continue
                    parent[sample_index] = neighbor
                    g[sample_index] = cost_through_neighbor
                    output.edge_indices.append([nearest_node,neighbor])
                    output.edge_weights.append(neighbor_dist)

        # check if we reached the goal. If not return empty path
        path = []
        if np.allclose(output.nodes[-1],end_node):
            current_node = len(output.nodes)-1
            while current_node != 0:
                if current_node in path:
                    raise Exception(f"Cycle detected in path.path {path}")
                path.append(current_node)
                current_node = parent[current_node]
            path.append(0)
            path.reverse()
        output.nodes = np.array(output.nodes)
        output.edge_indices = np.array(output.edge_indices)
        output.edge_weights = np.array(output.edge_weights)
        return output,path

    @classmethod
    def from_prm(
        cls, 
        n_samples:int, 
        sampler_func:Callable,
        node_collision_func:Callable,
        neighborhood_radius:float,
        edge_collision_func:Optional[Callable]=None,
        distance_func:Callable[[np.ndarray,np.ndarray],float]=EUCLEDIAN_DISTANCE_FUNCTION,
        max_connections_per_node:Optional[int]=None):
        """
        Creates a probabilistic roadmap (PRM) 
        for more information about PRM: https://en.wikipedia.org/wiki/Probabilistic_roadmap

        args:
            n_samples: int
                the number of samples to generate
            sampler_func: Callable
                a function that generates a random sample
            node_collision_func: Callable
                a function that checks if a node is in collision
            neighborhood_radius: float
                the maximum distance to connect neighbors
            edge_collision_func: Optional[Callable]
                a function that checks if an edge is in collision
                we can sometimes omit this function if the neighborhood radius is small enough
            distance_func: Callable[[np.ndarray,np.ndarray],float]
                distance function to calculate the cost of traversing an edge
            max_connections_per_node: Optional[int]
                the maximum number of connections per node
        """
        with logging_redirect_tqdm(loggers=[log]):
            nodes = []
            log.info("Generating nodes")
            with tqdm(total=n_samples) as pbar:
                while len(nodes) < n_samples:
                    node = sampler_func()
                    if not node_collision_func(node):
                        nodes.append(node)
                        log.debug(f"found collision free node: {node}")
                        pbar.update(1)
            nodes = np.array(nodes)
            n_nodes = len(nodes)
            edges = {}
            edge_indices = []
            edge_weights = []
            log.info("Generating edges")
            for i in tqdm(range(n_nodes)):
                node_connections = 0
                node_neighbors = []
                for j in range(n_nodes):
                    if i == j:
                        continue
                    if (i,j) in edges or (j,i) in edges:
                        continue
                    dist = distance_func(nodes[i],nodes[j])
                    if dist <= neighborhood_radius:
                        node_neighbors.append((j,dist))
                # node_neighbors.sort(key=lambda x:x[1],reverse=True)
                log.debug(f"node {i} has {len(node_neighbors)} neighbors")
                for j,dist in node_neighbors:
                    if max_connections_per_node is not None and node_connections >= max_connections_per_node:
                        break
                    if edge_collision_func is not None:
                        if edge_collision_func(nodes[i],nodes[j]):
                            continue
                    log.debug(f"found collision free edge between nodes {[i,j]}")
                    edge_indices.append([i,j])
                    edge_weights.append(dist)
                    node_connections += 1
                log.debug(f"node {i} has {node_connections} new connections")
            nodes = np.array(nodes)
            edge_indices = np.array(edge_indices)
            edge_weights = np.array(edge_weights)
        return cls(nodes,edge_indices,edge_weights)
    
    def save(self,save_path)->None:
        """
        Save the CollisionMap to a file
        """
        np.savez(save_path,nodes=self.nodes,edge_indices=self.edge_indices,edge_weights=self.edge_weights)

    @classmethod
    def load(cls,load_path)->'CollisionMap':
        """Load map from file"""
        data = np.load(load_path)
        return cls(data['nodes'],data['edge_indices'],data['edge_weights'])   

    def plot(self,show:bool=False, node_path:Optional[List[int]]=None)->Tuple[plt.Figure,plt.Axes]:
        """
        Plot the CollisionMap. Only works for 2D maps
        """
        # make sure the nodes are 2D
        assert self.nodes.shape[1] == 2, "Only 2D nodes are supported"
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        for edge in self.edge_indices:
            start = self.nodes[edge[0]]
            end = self.nodes[edge[1]]
            ax.plot([start[0],end[0]],[start[1],end[1]],'b', linewidth=0.35)
        for node in self.nodes:
            ax.plot(node[0],node[1],'ro',markersize=1)
        if node_path is not None:
            for i in range(len(node_path)-1):
                start = self.nodes[node_path[i]]
                end = self.nodes[node_path[i+1]]
                ax.plot([start[0],end[0]],[start[1],end[1]],'r', linewidth=1)
        if show:
            plt.show()
        return fig, ax
