#import sqlite3
import matplotlib.pyplot as plt
import os 
import pandas as pd 
import math

from vectorizedMethod import calculationOfRadiusAndCenters



def calculationOfRadiusAndCenters(file_path):
    listOfRadius = []
    centers = []
    diameters = []
    for fp in file_path:
        df = pd.read_csv(fp)
  
        for index, row in df.iterrows():
            x1 = row['left_boundary_x']
            x2 = row['right_boundary_x']
            y1 = row['left_boundary_y']
            y2 = row['right_boundary_y']
            
            diameter = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            diameters.append(diameter)
            radius = diameter / 2
            listOfRadius.append(radius)
 
            center_x = (x1+x2)/2
            center_y = (y1+y2)/2
            centers.append((center_x, center_y))
            
    return listOfRadius, centers

def plotRadiusAndCenters(radius, centers):
   
    center_x = [c[0] for c in centers]
    center_y = [c[1] for c in centers]

   
    plt.figure(figsize=(10, 5))


    plt.subplot(1, 2, 1)
   
    #plt.scatter(center_x, center_y, edgecolors='pink') 
    plt.scatter(center_x, center_y, edgecolors='pink')  # Tomma punkter


    plt.title('Centers (X, Y)')
    plt.xlabel('Center X')
    plt.ylabel('Center Y')
    plt.grid(True)
    

    
    plt.subplot(1, 2, 2)
    plt.plot(radius, color='purple')
    plt.title('Radius')
    plt.xlabel('Index')
    plt.ylabel('Radius')
    plt.grid(True)

    
    plt.figure(figsize=(8, 5))
    plt.boxplot(radius, vert=False)
    plt.title('Boxplot of Radius')
    plt.xlabel('Radius')
    plt.grid(True)



    
    plt.tight_layout()
    plt.show()



#dir = '/home/user/sense-environment/road-networks/asta_zero/fh_6x4_tractor_trailer/sil_map/segments'
dir = '/home/user/Desktop/newSegmentsFile'
file_paths = [os.path.join(dir,file) for file in os.listdir(dir) if file.endswith('.csv')]

radi, centers = calculationOfRadiusAndCenters(file_paths)
plotRadiusAndCenters(radi, centers)




def calculationOfRadiusAndCenters2(file_path):
    listOfRadius = []
    centers = []
    diameters = []

    df = pd.read_csv(file_path)
  
    for index, row in df.iterrows():
        diameter = row['Dimensions X']
        center_x = row['Center X']
        center_y = row['Center Y']

        #diameters.append(diameter)
        radius = diameter / 2
        listOfRadius.append(radius)

        centers.append((center_x, center_y))
        print(f'print centers: {centers}')

    return listOfRadius, centers

def plotRadiusAndCenters(radius, centers):
    
    center_x = [c[0] for c in centers]
    center_y = [c[1] for c in centers]

  
    plt.figure(figsize=(10, 5))

   
    plt.subplot(1, 2, 1)
    plt.scatter(center_x, center_y, color='pink')
    plt.title('Centers (X, Y)')
    plt.xlabel('Center X')
    plt.ylabel('Center Y')
    plt.grid(True)

    
    plt.subplot(1, 2, 2)
    plt.plot(radius, color='purple')
    plt.title('Radius')
    plt.xlabel('Index')
    plt.ylabel('Radius')
    plt.grid(True)

    plt.figure(figsize=(8, 5))
    plt.boxplot(radius, vert=False)
    plt.title('Boxplot of Radii')
    plt.xlabel('Radius')
    plt.grid(True)
    

    plt.tight_layout()
    plt.show()



csvFileFromBlender = '/home/user/Desktop/Circles.csv'
radius, centers = calculationOfRadiusAndCenters2(csvFileFromBlender)


plotRadiusAndCenters(radius, centers)
