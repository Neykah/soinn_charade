"""
Apply the SOINN algorithm to a set of N body pose estimations
chosen randomly in the Charades/keypoints directory.

Usage: ./train_charades.py [N_train] [threshold] [delete_node period] [max_edge_age]

Author: Morgan Lefranc
"""

import sys
import os
import random
import joblib
import numpy as np
import cv2

sys.path.append('/home/morgan/soinn/soinn')
import openpose_yaml
from soinn import Soinn




def normalize(vector):
    yI = 0.5 * (vector[17] + vector[23])
    Nx = abs(vector[10]-vector[4]) / 100
    Ny = abs(vector[3]-yI) / 100
    N = 0.5 * (Nx + Ny)

    result = []
    for i, elem in enumerate(vector):
        elem /= N
        if i == 0:
            kx = 600 - elem
        elif i == 1:
            ky = 100 - elem
        if i%2 == 0:
            elem += kx
        else:
            elem += ky
        result.append(elem)
    return result

def is_reliable(vector, T):
    """
    For a given vector T=(x, y, confidence_score), check if
    points 0 1 2 5 8 11 exist (mandatory for normalizing task !)
    and if their average confidence score > T.
    If so, check if the mean confidence score of the vector
    is > to T. If so, return True. Else, return False.
    """
    confidence = [e for i, e in enumerate(vector) if i % 3 == 2]
    mean_conf = np.mean(confidence)
    crit_conf = [confidence[i] for i in [0, 1, 2, 5, 8, 11]]
    mean_crit_conf = np.mean(crit_conf)
    if 0 not in crit_conf:
        if mean_crit_conf >= T:
            return mean_conf >= T
        return False
    return False

def delete_confidence(vector):
    return [e for i, e in enumerate(vector) if i % 3 != 2]

def prepare_dataset(N_train, T):
    """ Create a new dataset of N_train frames for which quality is above T.
    """
    print('load Charades/Openpose dataset')
    charades_poses = np.array([])
    keypoints_dir = '/media/morgan/HDD/Aolab/Data/Charades_keypoints/'
    index_file_name = '/media/morgan/HDD/Aolab/Data/index_poses'
    openpose_yaml.add_constr_repr()
    N = 28
    video_used = []

    with open(index_file_name, 'r') as f:
        i = 0 # Number of frames currently in the dataset.
        lines = f.read().splitlines()
        while i < N_train:
            line = random.choice(lines)
            print(line)
            # Example : CBPJF_000000000016
            title = line[:18]
            headtitle = title[:5]
            # Example : /media/morgan/HDD/Aolab/Data/Charades_keypoints/CBPJF/CBPJF_000000000016_pose.yml
            file_name = keypoints_dir + title[:5] + '/' + line.strip('\n')
            print(i)
            # Open the corresponding yaml file.
            signals_input = openpose_yaml.load_yaml(file_name)
            signals_list_42 = [s for s in signals_input if is_reliable(s, T)]
            signals_list = [delete_confidence(s) for s in signals_list_42]

            if len(signals_list) == 1: # If there is one person on the scene
                n = charades_poses.shape[0]
                charades_poses.resize(n+1, N)
                charades_poses[-1, :] = normalize(signals_list[0])
                #~ charades_poses[-1,:] = signals_list[0]
                i += 1
                if headtitle not in video_used:
                    video_used.append(headtitle)
            elif len(signals_list) > 1: # If there are more than 1 person on the scene
                s = len(signals_list)
                n = charades_poses.shape[0]
                charades_poses.resize(n+s, N)
                for j in range(s):
                    charades_poses[n+j, :] = normalize(signals_list[j])
                    #~ charades_poses[n+j, :] = signals_list[j]
                    i += s
                if headtitle not in video_used:
                    video_used.append(headtitle)
    return (charades_poses, video_used)

def learning(soinn, x_train):
    for data_x in x_train:
        soinn.input_signal(data_x)

def draw_pose(body_parts, im_write=False, im_number=0, im_show=False):
    h, w = 1200, 1200
    body_parts = np.floor(body_parts).astype(int)
    img = np.zeros((h, w, 3))
    points = []
    N = 28
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(0, N, 2):
        points.append((body_parts[i], body_parts[i+1]))
        if points[-1] > (50,50):
            cv2.circle(img, points[-1], 5, (0, 0, 255), -1)
            cv2.putText(img, str(int(i/2)), points[-1], font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    articulations = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                     (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)]
    for (i, j) in articulations:
        if points[i] > (50,50) and points[j] > (50,50):
            cv2.line(img,points[i], points[j], (255,0,0), 2)

    # Print the image
    if im_write:
        cv2.imwrite("cluster_{}.png".format(im_number), img)
    if im_show:
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# If a SOINN file already exists: load it
# Else, create a new SOINN model for data with specified parameters
def main(argv):
    """
    Main function, check module docstring.
    """
    (N_TRAIN, T, delete_node_period, max_edge_age) = (1000, 0.6, 100, 30)
    if len(argv) >= 2:
        N_TRAIN = int(argv[1])
    if len(argv) >= 3:
        T = int(argv[2])
    if len(argv) >= 4:
        delete_node_period = int(argv[3])
    if len(argv) >= 5:
        max_edge_age = int(argv[4])
    if len(argv) >= 6:
        print("""Usage: ./train_charades.py [N_train] [threshold]
              [delete_node period] [max_edge_age]""")
        raise IndexError

    im_write = True
    dumpfile = 'outputs/soinn{0}_{1}_{2}_{3}.dump'.format(N_TRAIN, delete_node_period, max_edge_age, int(100*T))

    # If a SOINN node already exist for those parameters, use it.
    try:
        soinn_i = joblib.load(dumpfile)

    except FileNotFoundError:
        # If no SOINN already exist, define the dataset according to SOINN parameters
        dataset, video_used = prepare_dataset(N_TRAIN, T)
        with open('video_used_{}_{}'.format(N_TRAIN, int(100*T)), 'w') as f:
            for title in video_used:
                f.write('{}\n'.format(title))
        print('New SOINN is created.')
        soinn_i = Soinn(delete_node_period=delete_node_period, max_edge_age=max_edge_age)
        learning(soinn_i, dataset)
    soinn_i.print_info()
    soinn_i.save(dumpfile)
    if im_write:
        dir_name = "clusters_{0}_{1}_{2}_{3}".format(N_TRAIN, delete_node_period, max_edge_age, int(100*T))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        os.chdir(dir_name)
    for i, node in enumerate(soinn_i.nodes):
        print(node)
        draw_pose(node, im_write, i, im_show=False)

if __name__ == "__main__":
    main(sys.argv)
