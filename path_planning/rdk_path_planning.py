from path_planning.configuration_space_map import CollisionMap
import numpy as np
from robodk import robolink
from typing import List, Optional
from contextlib import contextmanager

@contextmanager
def rdk_collision_mapping_mode(rdk:robolink.Robolink=None):
    """
    Context manager to enable collision mapping mode in RoboDK
    In collision mapping mode, we enable collision checking and disable rendering
    """
    if rdk is None:
        rdk = robolink.Robolink()
    try:
        rdk.setCollisionActive(robolink.COLLISION_ON)
        rdk.Render(False)
        yield
    finally:
        rdk.setCollisionActive(robolink.COLLISION_OFF)
        rdk.Render(True)

# configuration distance functions
def eucledean_distance(J1:List[float],J2:List[float],weights:Optional[List[float]]=None)->float:
    diff = np.array(J1)-np.array(J2)
    if weights is not None:
        diff = diff*np.array(weights)
    return np.linalg.norm(diff)

def manhattan_distance(J1:List[float],J2:List[float],weights:Optional[List[float]]=None)->float:
    diff = np.array(J1)-np.array(J2)
    if weights is not None:
        diff = diff*np.array(weights)
    return np.sum(np.abs(diff))

def chebyshev_distance(J1:List[float],J2:List[float],weights:Optional[List[float]]=None)->float:
    diff = np.array(J1)-np.array(J2)
    if weights is not None:
        diff = diff*np.array(weights)
    return np.max(np.abs(diff))

class RDKPathPlanner:
    def __init__(
            self,
            rdk:Optional[robolink.Robolink]=None,
            robot_name:Optional[str]=None,
            min_step:float=4.0,
            collision_map:Optional[CollisionMap]=None,
            configuration_distance_func:callable=eucledean_distance,
            ):        
        if rdk is None:
            rdk = robolink.Robolink()
        self.rdk = rdk
        if robot_name is None:
            self.robot = rdk.ItemUserPick('Select a robot', robolink.ITEM_TYPE_ROBOT)
        else:
            self.robot = rdk.Item(robot_name, robolink.ITEM_TYPE_ROBOT)
            assert self.robot.Valid(),"Robot not found"
        self.min_step = min_step
        # get joint limits
        lower_joint_limits, upper_joint_limits,_ = self.robot.JointLimits()
        self._lower_joint_limits = lower_joint_limits.tolist()
        self._upper_joint_limits = upper_joint_limits.tolist()

        self.collision_map = collision_map
        self.configuration_distance_func = configuration_distance_func

    def _random_robot_configuration(self)->List[float]:
        """Return a random joint configuration for the robot"""
        return np.random.uniform(self._lower_joint_limits, self._upper_joint_limits).tolist()
    
    def _configuration_collision_check(self,configuration:List[float])->bool:
        """Check if the robot is in collision at the given configuration
        returns True if no collision is detected
        """
        self.robot.setJoints(list(configuration))
        self.rdk.Update()
        return self.rdk.Collisions()>0
    
    def _move_collision_check(self,J1:List[float],J2:List[float])->bool:
        """
        Check if the robot is in collision when moving from J1 to J2
        returns True if no collision is detected
        """
        return self.robot.MoveJ_Test(list(J1),list(J2)) != 0
    
    def _path_collision_check(self,path:List[List[float]])->bool:
        """
        Check if the robot is in collision when moving along the given path
        returns True if no collision is detected
        """
        for i in range(len(path)-1):
            if not self._move_collision_check(path[i],path[i+1]):
                return False
        return True
    
    def _random_valid_configuration(self)->List[float]:
        """Return a random valid configuration for the robot"""
        while True:
            config = self._random_robot_configuration()
            if not self._configuration_collision_check(config):
                return config
            
    def generate_prm(
            self, 
            n_samples:int, 
            neighborhood_radius:float,
            max_connections_per_node:Optional[int]=None)->None:
        """
        Generate a Probabilistic Roadmap (PRM) for the robot
        """
        with rdk_collision_mapping_mode(self.rdk):
            self.collision_map = CollisionMap.from_prm(
                n_samples=n_samples,
                sampler_func=self._random_robot_configuration,
                node_collision_func=self._configuration_collision_check,
                neighborhood_radius=neighborhood_radius,
                edge_collision_func=self._move_collision_check,
                distance_func=self.configuration_distance_func,
                max_connections_per_node=max_connections_per_node,
            )

    def generate_rrt_plan(
            self,
            start_configuration:List[float],
            goal_configuration:List[float],
            max_iterations:int=100,
            goal_bias:float=0.1,
            step_size:float=4.0,
            )->List[List[float]]:
        """
        Generate a path from start to goal using the RRT algorithm
        """
        with rdk_collision_mapping_mode(self.rdk):
            rrt_map,plan = CollisionMap.from_rrt(
                n_samples=max_iterations,
                sampler_func=self._random_robot_configuration,
                step_size=step_size,
                node_collision_func=self._configuration_collision_check,
                start_node=np.array(start_configuration),
                end_node=np.array(goal_configuration),
                solution_atol=step_size,
                end_node_check_ratio=goal_bias,
                distance_func=self.configuration_distance_func,
                edge_collision_func=self._move_collision_check,
            )
        self.collision_map = rrt_map
        if len(plan)>0:
            return [start_configuration]+[self.collision_map.nodes[i].tolist() for i in plan]+[goal_configuration]
        return []

    def plan_path(
            self,
            start_configuration:List[float],
            goal_configuration:List[float],
            )->List[List[float]]:
        """
        Plan a path from start to goal using the PRM
        """
        assert self.collision_map is not None, "PRM not generated"
        start_node = self.collision_map.nearest_neighbor(np.array(start_configuration))
        goal_node = self.collision_map.nearest_neighbor(np.array(goal_configuration))
        path_indices,cost = self.collision_map.a_star(start_node,goal_node,node_distance_function=self.configuration_distance_func)
        print(path_indices)
        output = [start_configuration]+[self.collision_map.nodes[i].tolist() for i in path_indices] + [goal_configuration]
        return output
