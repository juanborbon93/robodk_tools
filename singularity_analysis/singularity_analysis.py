"""
This module provides functions to evaluate the manipulability of a RoboDK program.
It calculates the manipulability of each move instruction in the program and plots the results.
If the robot approaches a joint singularity, the manipulability score will approach zero.
"""

from pybotics.robot import Robot
from pybotics.tool import Tool
from pathlib import Path
import numpy as np
from robodk import robolink, robomath
from pydantic import BaseModel
from typing import List
import pandas as pd
from matplotlib import pyplot as plt

class RobotManipulability(BaseModel):
    name: str
    is_joint_move: bool
    value: float

class ProgramManipulability(BaseModel):
    angular: List[RobotManipulability] = []
    cartesian: List[RobotManipulability] = []


    @property
    def angular_df(self):
        return pd.DataFrame.from_records([x.model_dump() for x in self.angular])

    @property
    def cartesian_df(self):
        return pd.DataFrame.from_records([x.model_dump() for x in self.cartesian])
    
    def plot_angular(self):
        pass

    def plot_cartesian(self):
        pass

DH_MODEL_FILE = Path(__file__).parent / "robot_dh_model.txt"

def get_dh_robot_from_file(rdk_robot:robolink.Item):
    """
    Load the DH robot model from a file and set the tool for the robot.

    Args:
        rdk_robot (robolink.Item): The robot item in RoboDK.

    Returns:
        pybotics.robot.Robot: The DH robot model.
    """
    dh_data = np.loadtxt(DH_MODEL_FILE, delimiter=',', skiprows=1)
    dh_robot = Robot.from_parameters(dh_data.T)

    # set tool for dh_robot to match the robot tool in RoboDK
    dh_tool = Tool()
    dh_tool.matrix = rdk_robot.PoseTool().toNumpy()
    dh_robot.tool = dh_tool

    # check that our kinematic model is accurate
    rdk_joints = rdk_robot.Joints().tolist()
    original_pose_frame = rdk_robot.PoseFrame()
    try:
        rdk_robot.setPoseFrame(robomath.eye(4))
        tx_rdk_robot_base_flange = rdk_robot.Pose().toNumpy()
    finally:
        rdk_robot.setPoseFrame(original_pose_frame)
    
    # WARNING: we convert all joint values to radians. This could be an issue if your robot has a primsatic joint
    dh_joints = np.deg2rad(rdk_joints)
    tx_dh_robot_base_flange = dh_robot.fk(dh_joints)
    if not np.allclose(tx_rdk_robot_base_flange, tx_dh_robot_base_flange, atol=1e-6):
        print("Warning: DH model does not match RoboDK robot")
        print("RoboDK robot base to flange transformation:")
        print(tx_rdk_robot_base_flange)
        print("DH robot base to flange transformation:")
        print(tx_dh_robot_base_flange)
        instructions = f"""
        To update the DH model:
        1. Open the robot panel in RoboDK
        2. Open the parameters menu 
        3. Click on "Export Table"
        4. Select the "Denavit Hartenberg Modified (DHM)" option
        5. Click on "copy data"
        6. Paste the data in the file {DH_MODEL_FILE}
        """
        print(instructions)
        raise ValueError("DH model does not match RoboDK robot")

    return dh_robot

def evaluate_rdk_program(program_name):
    """
    Evaluate a RoboDK program by calculating the manipulability of each move instruction.

    Args:
        program_name (str): The name of the program to evaluate. If None, a program selection dialog will be shown.

    Raises:
        AssertionError: If the program or robot is not found.
    """
    rdk = robolink.Robolink()
    if program_name is None:
        rdk_prog = rdk.ItemUserPick("Select a program to evaluate", robolink.ITEM_TYPE_PROGRAM)
        
    else:
        print(f"Loading program {program_name}")
        rdk_prog = rdk.Item(program_name,robolink.ITEM_TYPE_PROGRAM)
        assert rdk_prog.Valid(), f"Program {program_name} not found"
    rdk_robot = rdk_prog.getLink(robolink.ITEM_TYPE_ROBOT)
    assert rdk_robot.Valid(), f"Robot not found in program {program_name}"
    dh_robot = get_dh_robot_from_file(rdk_robot)

    msg, program_joints, status  = rdk_prog.InstructionListJoints(deg_step=0.25)
    program_joints = program_joints.toNumpy()

    move_instructions = []
    names = []
    for i in range(rdk_prog.InstructionCount()):
        name,ins_type,move_type,is_joint_target,target, joints= rdk_prog.Instruction(i)
        names.append(name)
        name_repeats = names.count(name)
        if name_repeats > 1:
            name = name + f" {name_repeats}"

        if move_type:
            not_joint_move = move_type != robolink.MOVE_TYPE_JOINT
            move_instructions.append((name,not_joint_move))

    manipulability = ProgramManipulability()
    for joints in program_joints.T:
        move_id = int(joints[-1])-1
        instruction_name,is_joint_move = move_instructions[move_id]
        joints_radians = np.deg2rad(joints[0:dh_robot.ndof])
        jacobian = dh_robot.jacobian_world(joints_radians)
        # split the jacobian into angular and cartesian components
        jacobian_angular = jacobian[:3,:]
        jacobian_cartesian = jacobian[3:,:]
        # calculate the manipulability ellipsoid
        A_cartesian = jacobian_cartesian @ jacobian_cartesian.T
        A_angular = jacobian_angular @ jacobian_angular.T
        # calculate the eigenvalues of the manipulability ellipsoid
        # these eigen values correspond to the lengths of the principal axes of the ellipsoid
        cartesian_eigen_values = np.linalg.eigvals(A_cartesian)
        angular_eigen_values = np.linalg.eigvals(A_angular)
        # ratio of the smallest to the largest eigenvalue
        cartesian_ratio_manipul = np.sqrt(np.min(cartesian_eigen_values) / np.max(cartesian_eigen_values))
        angular_ratio_manipul = np.sqrt(np.min(angular_eigen_values) / np.max(angular_eigen_values))
        manipulability.angular.append(
            RobotManipulability(name=instruction_name,is_joint_move=is_joint_move,value=angular_ratio_manipul)
        )
        manipulability.cartesian.append(
            RobotManipulability(name=instruction_name,is_joint_move=is_joint_move,value=cartesian_ratio_manipul)
        )

    # show plots
    ax = plt.subplot(2,1,1)
    df = manipulability.cartesian_df
    for name, group in df.groupby('name'):
        ax.plot(group.value, label=name)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("Cartesian manipulability")
    ax.set_ylabel("Manipulability")
    # set vertical space between the two plots
    plt.subplots_adjust(hspace=0.5)
    ax = plt.subplot(2,1,2)
    df = manipulability.angular_df
    for name, group in df.groupby('name'):
        ax.plot(group.value, label=name)
    ax.set_title("Angular manipulability")

    ax.set_ylabel("Manipulability")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-name", help="Name of the program to evaluate",default=None)
    args = parser.parse_args()
    evaluate_rdk_program(args.program_name)




