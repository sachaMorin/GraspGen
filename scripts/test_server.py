# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import glob
import json
import os

import numpy as np
import requests

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load sample data from json to test GraspGen server"
    )
    parser.add_argument(
        "--sample_data_dir",
        type=str,
        default="",
        help="Directory containing JSON files with point cloud data",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5005,
        help="Port for the server",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.sample_data_dir == "":
        raise ValueError("sample_data_dir is required")

    if not os.path.exists(args.sample_data_dir):
        raise FileNotFoundError(
            f"sample_data_dir {args.sample_data_dir} does not exist"
        )

    json_files = glob.glob(os.path.join(args.sample_data_dir, "*.json"))

    for json_file in json_files:
        print(json_file)

        data = json.load(open(json_file, "rb"))

        obj_pc = np.array(data["object_info"]["pc"])
        obj_pc_color = np.array(data["object_info"]["pc_color"])
        grasps = np.array(data["grasp_info"]["grasp_poses"])
        grasp_conf = np.array(data["grasp_info"]["grasp_conf"])

        full_pc_key = "pc_color" if "pc_color" in data["scene_info"] else "full_pc"
        xyz_scene = np.array(data["scene_info"][full_pc_key])[0]
        xyz_scene_color = np.array(data["scene_info"]["img_color"]).reshape(1, -1, 3)[
            0, :, :
        ]

        obj_xyzrgb = np.hstack((obj_pc, obj_pc_color))
        scene_xyzrgb = np.hstack((xyz_scene, xyz_scene_color))


        # Request
        data = [obj_xyzrgb.tolist(), scene_xyzrgb.tolist()]
        payload = {"obj_scene_xyzrgb": data}

        resp = requests.post(f"http://{args.host}:{args.port}/graspgen", json=payload)
        resp.raise_for_status()
        data = resp.json()

        grasps, grasp_conf = data["grasps"], data["grasp_conf"]
        
        print(f"Received {len(grasps)} grasps from server.")

        if len(grasps) > 0:

            input("Press Enter to continue to next scene...")
        else:
            print("No grasps found! Skipping to next scene...")
