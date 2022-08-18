import json
import pandas as pd
import numpy as np
import glob
import csv

def getFileName(path):
    filelist = glob.glob(path + "/*")
    return filelist

def getSpecificData(filepath):
    
    with open(filepath) as f:
        data = json.load(f)
        data = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1,3)
    df = pd.DataFrame(data, columns=['X','Y','P'], index=["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", \
        "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"])

    # 自分の必要なデータを取り出す
    writeCSV([float(df.at["Neck", "X"]), float(df.at["Neck", "Y"]), float(df.at["MidHip", "X"]), float(df.at["MidHip", "Y"])])

def writeCSV(data):
    with open('output.csv', 'a') as f:
        writer = csv.writer(f, lineterminator='\n') 
        writer.writerow(data)

def main():
    
    
    with open('output.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n') 
        writer.writerow(["Neck_x","Neck_y","MidHip_x","MidHip_y"])
    filepath = "./input/keypoint.json"
    getSpecificData(filepath)

if __name__ == '__main__':
    main()
