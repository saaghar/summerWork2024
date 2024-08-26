import glob
import random
import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot
import csv
import skimage as si
import os
import time
import tables as tb
from databaseBuilder import createDatabase, insertDatabase


#Calculation of radius and centers from the segment files 
def calculationOfRadiusAndCenters(file_path):
    listOfRadius = []
    centers = []
    diameters= []
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
 
#Calculation of radius and centers from the CSV file performed in Blender.  
def centersAndRadiusFromBlender(file_path):
    listOfRadius = []
    centers = []

    df = pd.read_csv(file_path)
  
    for index, row in df.iterrows():
        diameter = row['Dimensions X']
        center_x = row['Center X']
        center_y = row['Center Y']

        #diameters.append(diameter)
        radius = diameter / 2
        listOfRadius.append(radius)

        centers.append((center_x, center_y))

    return listOfRadius, centers




#matrix rotation 
def rotatePoints(coord_list, ego_pos, angle, direction="clockwise", mode="global"):
    """
    :param coord_list: list of (x,y) tuples containing coordinates
    :param ego_pos: if local mode: point to rotate around. if global mode, position of ego vehicle.
    :param angle: rotation angle in radians
    :param direction: clockwise or anti-clockwise rotation
    :param mode: rotation around origin (global) or around (x,y) (local).
    :return: list of (x',y') tuples containing rotated points around Origin.
    """
    if direction in ("clockwise", "CW"):
        r_mat = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    elif direction in ("anti-clockwise", "counter-clockwise", "CCW"):
        r_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    else:
       sys.exit(1)
    
    # rotate list of points from (x,y) coordinates to (x', y') rotated coordinates
    coord_list = np.array(coord_list, dtype=float)
    if mode == "global":
        for i in range(len(coord_list)):
            coord_list[i] = np.matmul(r_mat, coord_list[i])
        # rotate ego pos from (x,y) to (x', y')
        ego_pos = np.matmul(r_mat, np.array(ego_pos))
        return coord_list, ego_pos
    
    if mode == "local":
        coord_list = coord_list - ego_pos
        for i in range(len(coord_list)):
            coord_list[i] = np.matmul(r_mat, coord_list[i])  
        coord_list = coord_list + ego_pos
        return coord_list, ego_pos
 
 
 
#Build an ogrid and adjust parameter according to input data
def revampedGridBuild(h5_path, ego_pos: np.ndarray, db_resolution, grid_dimensions: np.ndarray, resolution, listOfRadius, centers, cell_size, ego_yaw, direction):
    #record start time
    start_time = time.time()

    offset_start_time = time.time()
    offset = grid_dimensions // 2
    offset_end_time = time.time()
    print(f"Time taken for offset calculation: {offset_end_time - offset_start_time}")
    print(f"offset inside revampedGridBuild: {offset}")
    
    
    #adjust all parameters based on db_resolution
    if db_resolution < 1:
        ego_pos = (ego_pos / db_resolution).astype(int)
        grid_dimensions = (grid_dimensions / db_resolution).astype(int)
 
    
    #create ogrid
    ogrid = np.zeros([grid_dimensions[1], grid_dimensions[0]], dtype=int)
 
    #Rotate circle centers and ego position around origin
    print(f"before rot ego pos: {ego_pos}")
    rotate_start_time = time.time()
    adjusted_centers, ego_pos = rotatePoints(centers, ego_pos, ego_yaw, "CW", mode='global')
    rotate_end_time = time.time()
    print(f"Time taken to rotate points: {rotate_end_time - rotate_start_time:.4f} seconds")

    #Adjust ego position to the center of the grid
    print(f"Ego position: {ego_pos}")
    adjusted_ego_pos, offset_x, offset_y = adjustEgoPosition(ego_pos, grid_dimensions[0], grid_dimensions[1], cell_size)
    print(f"New ego position: {adjusted_ego_pos}, Offsets: ({offset_x}, {offset_y})")
 
    # Adjust circle centers based on the new ego position and offset
    adjusted_centers = adjustCirclesCenter(adjusted_centers, offset_x, offset_y, cell_size)
 
    # Scale radius
    scaled_radius = [radius / cell_size for radius in listOfRadius]
 
    # Draw circles on the grid
    step_start_time = time.time()
    ogrid = drawCircles_2(ogrid, adjusted_centers, scaled_radius)
    
    ogrid = np.rot90(ogrid, 3)
    ogrid = np.flipud(ogrid)
    
 
    step_end_time = time.time()
    print(f"Time taken for drawing circles: {step_end_time - step_start_time}")
 
    total_time = time.time() - start_time
    print(f"Total time for revampedGridBuild: {total_time}")
    
    return ogrid, adjusted_ego_pos
    
 
 
#read ego_pos and yaw from h5file
def readDataFromH5(h5_path):
    with tb.open_file(h5_path, mode='r') as h5file:
        table = h5file.get_node("/", "LocalizationOutput")

        ego_pos = np.array([[row['ego_pos'][0], row['ego_pos'][1]] for row in table])
        ego_yaw = np.array([row['ego_yaw'] for row in table])    
    return ego_pos, ego_yaw
 
 
#Read surface belief from h5file
def readSurfaceBelief(h5_path):
    with tb.open_file(h5_path, mode='r') as h5file:
        table = h5file.get_node("/", "PerceptionOutput")
        surface_belief = table.cols.surface_belief[:]
    return surface_belief  
 
 
#Move the road to the first district in coordinate system
def adjustCirclesCenter(centers, grid_width, grid_height, cell_size):
    print(centers)
    adjustedCenters = [(x / cell_size + grid_width, y / cell_size + grid_height) for x, y in zip(centers[:, 0], centers[:, 1])]
    return adjustedCenters
 
 
 
 
 
#calculation of ego position and offset
def adjustEgoPosition(ego_pos, grid_width, grid_height, cell_size):
    scaled_ego_pos = np.array(ego_pos)/cell_size
 
    # offset-> bring ego_pos to the center of ogrid
    offset_x = grid_width / 2 - scaled_ego_pos[0]
    offset_y = grid_height / 2 - scaled_ego_pos[1]
    
    #New ego position is in the center of the grid
    adjustedEgoPos = (grid_width/ 2, grid_height / 2)
    return adjustedEgoPos, offset_x, offset_y
 
 

 
#calculate the distance from each point to the center
def drawCircles_1(grid, center, radius):
    center_x, center_y = center
    
    if 0 <= center_x < grid.shape[0] and 0 <= center_y < grid.shape[1]:
        y, x = np.ogrid[:grid.shape[0], :grid.shape[1]]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        grid[dist_from_center <= radius] = 1
    else:
        y, x = np.ogrid[:grid.shape[0], :grid.shape[1]]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        grid[dist_from_center <= radius] = 1
 
#draw the circles
def drawCircles_2(grid, centers, radius):
    for center, radius in zip(centers, radius):
        drawCircles_1(grid, center, radius)
    return grid
 
 
 
#The methods below are written to test things work as they should
def testOffsetAndEgo(ego_pos, grid_width, grid_height, cell_size=0.25):
  
   adjusted_ego_pos, offset_x, offset_y = adjustEgoPosition(ego_pos, grid_width, grid_height, cell_size)
 
 
   scaled_ego_pos = np.array(ego_pos)/cell_size
 
 
   expected_offset_x = grid_width/2 - scaled_ego_pos[0]
   expected_offset_y = grid_height/2 - scaled_ego_pos[1]
   expected_adjustedEgoPos = (grid_width / 2, grid_height / 2)
 
 
   offset_x_correct = np.isclose(offset_x, expected_offset_x)
   offset_y_correct = np.isclose(offset_y, expected_offset_y)
   adjustedEgoPos_correct = np.isclose(adjusted_ego_pos, expected_adjustedEgoPos)
  
   test_result = {
       'Original Ego Position (meters)': ego_pos,
       'Scaled Ego Position (cells)': scaled_ego_pos,
       'Calculated Offset X': offset_x,
       'Expected Offset X': expected_offset_x,
       'Offset X Correct': offset_x_correct,
       'Calculated Offset Y': offset_y,
       'Expected Offset Y': expected_offset_y,
       'Offset Y Correct': offset_y_correct,
       'Adjusted ego position (orginal)': adjusted_ego_pos,
       'Adjusted ego poisition correct' : adjustedEgoPos_correct
   }
  
   return pd.DataFrame([test_result])
 
# grid_width = 400  # Grid width in cells
# grid_height = 200  # Grid height in cells

def testCirclesAndScaling(file_path, ego_pos, db_resolution, grid_dimensions, resolution, cell_size):
   listOfRadius, centers = calculationOfRadiusAndCenters(file_path)

  
   scaled_radius = [radius / cell_size for radius in listOfRadius]
   adjusted_ego_pos, offset_x, offset_y = adjustEgoPosition(ego_pos, grid_dimensions[0], grid_dimensions[1], cell_size)
   adjusted_centers = adjustCirclesCenter(centers, offset_x, offset_y)
  
   test_results = []
  
   for original_radius, scaled_r, center, adj_center in zip(listOfRadius, scaled_radius, centers, adjusted_centers):
       original_diameter = original_radius * 2
       scaled_diameter = scaled_r * 2
       test_results.append({
           'Original Radius (meters)': original_radius,
           'Scaled Radius (cells)': scaled_r,
           'Original Diameter (meters)': original_diameter,
           'Scaled Diameter (cells)': scaled_diameter,
           'Original Center': center,
           'Adjusted Center': adj_center
       })
  
   results_df = pd.DataFrame(test_results)
   return results_df
  
def findBoundaries(file_paths):
    boundaries = []
    
    for fp in file_paths:
        df = pd.read_csv(fp)
 
        for index, row in df.iterrows():
            x1 = df['left_boundary_x']
            x2 = df['right_boundary_x']
            y1 = df['left_boundary_y']
            y2 = df['right_boundary_y']
 
            boundaries.append(((x1,x2),(y1,y2)))  
 
    return boundaries
 
 
 
 
def main():
    start_main_time = time.time()

    #Plot
    h5_path = '/home/user/Downloads/SUT_275729_lunch_data/TEXL-001/2024-04-30_09-15-13_output.h5'
    dir = '/home/user/sense-environment/road-networks/asta_zero/fh_6x4_tractor_trailer/sil_map/segments'
    file_paths = [os.path.join(dir,file) for file in os.listdir(dir) if file.endswith('.csv')]
    csvFileFromBlender = '/home/user/Desktop/Circles.csv'
    
    
    #createDatabase()

    #calculate radius and centers from the file path
    radius_time_start = time.time()
    listOfRadius, centers = calculationOfRadiusAndCenters(file_paths)
    #listOfRadius , centers = centersAndRadiusFromBlender (csvFileFromBlender)
    radius_time_end = time.time()
    print(f"Time taken to calculate radius and centers: {radius_time_start - radius_time_end}")

    #insertDatabase(centers, listOfRadius, dataset_id=1)
    #insertDatabase(centers, listOfRadius, dataset_id=2)
 
    #Plot the ogrid 
    def plotOgrid(ogrid):
        fig, ax = plt.subplots()
        ax.imshow(ogrid, cmap='gray', interpolation='nearest')

        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.25)
        ax.grid(which='major', color='r', linestyle='-', linewidth=2)
 
        ax.set_xticks(np.arange(0, 200 + 0.25, 0.25), minor=True)  # 200*0.25 = 50 -> From 0 to 50 in steps of 0.25
        ax.set_yticks(np.arange(0, 400 + 0.25, 0.25), minor=True)  #400*0.25 = 100 -> From 0 to 100 in steps of 0.25
        ax.minorticks_on() 
 
        ax.scatter(ego_pos[0], ego_pos[1], s=10, c='pink')
 
        plt.imshow(ogrid, cmap='gray')
        plt.title("Ogrid")
        plt.xlabel("0 -> x")  
        plt.ylabel("y -> 0")
        plt.show()  
 
 
    def plotAllPoints(file_paths, ego_pos, ego_yaw):
        fig, ax = plt.subplots()
        
 
        for fp in file_paths:
            df = pd.read_csv(fp)
 
            x1 = df['left_boundary_x']
            x2 = df['right_boundary_x']
            y1 = df['left_boundary_y']
            y2 = df['right_boundary_y']
 
            p1 = [(x,y) for x,y in zip(x1, y1)]
            p2 = [(x,y) for x,y in zip(x2, y2)]
 
 
            p1, _ = rotatePoints(p1, ego_pos, ego_yaw, direction, mode='local')
            p2, _ = rotatePoints(p2, ego_pos, ego_yaw, direction, mode='local')
 
            p1 = np.array(p1)
            p2 = np.array(p2)
 
            # Plot the left boundary points
            ax.scatter(2*ego_pos[0]-p1[:,0], 2*ego_pos[1]-p1[:,1], c='blue', s=1, label='Left Boundary' if fp == file_paths[0] else "")
 
            # Plot the right boundary points
            ax.scatter(2*ego_pos[0]-p2[:,0], 2*ego_pos[1]-p2[:,1], c='red', s=1, label='Right Boundary' if fp == file_paths[0] else "")
        
        ax.scatter(ego_pos[0], ego_pos[1], s=10, c='pink')
 
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Boundary Points')
        plt.legend()
        plt.show()
 
 
 
 
 
    def plotSurfaceBeliefAsLine(ogrid):
        from matplotlib.colors import Normalize
 
        # surface_belief_matrix = readSurfaceBelief(h5_path)
        # slice_2d = surface_belief_matrix[:, :, 0]
        fig, ax = plt.subplots()
        x = np.arange(ogrid.shape[1])
        y = np.arange(ogrid.shape[0])
        xv, yv = np.meshgrid(x, y)
        norm = Normalize(vmin=ogrid.min(), vmax=ogrid.max())
        scatter = ax.scatter(xv, yv, c=ogrid.flatten(), cmap='gray', norm=norm)
 
        plt.title("Surface Belief Matrix as Scatter Plot")
        plt.xlabel("X Dimension")
        plt.ylabel("Y Dimension")
        cbar = plt.colorbar(scatter, ax=ax, label='Belief Value')
        plt.show()
 
   
    
    
    surface_belief_matrix = readSurfaceBelief(h5_path)
    db_resolution = 1
    grid_dimensions = np.array([200, 100]) 
    resolution = 1
    cell_size = 0.25
    direction = 'clockwise'
    direction_opp = 'anti-clockwise'
 

    data_load_start = time.time()
    ego_pos_list, ego_yaw_list = readDataFromH5(h5_path)
    data_load_end = time.time()
    print("f{Time taken to load data from h5: {data_load_end - data_load_start:.4f}")
 
    # read from h5-file 
    with tb.open_file(h5_path, mode='r+') as h5f:
        tbl = h5f.get_node("/", "PerceptionOutput")
        surface_ogrid_shape = tbl.cols.surface_ogrid[0].shape
 
    # Loop through ego_pos and ego_yaw to uppdate the ogrid 
        start_index = 100
        for i, (ego_pos, ego_yaw) in enumerate(zip(ego_pos_list[start_index:], ego_yaw_list[start_index:])):
            loop_start_time = time.time()

            ogrid, updated_ego_pos = revampedGridBuild(h5_path,ego_pos, db_resolution, grid_dimensions, resolution,listOfRadius, centers,cell_size, ego_yaw, direction)
 
            loop_end_time = time.time()
            print(f"Time taken for iteration {i + start_index}: {loop_end_time - loop_start_time}")

            #print(i+start_index)
            
  
            if i % 100 == 0: 
                plotOgrid(ogrid)
                plotAllPoints(file_paths, ego_pos, ego_yaw)
                plotSurfaceBeliefAsLine(tbl.cols.surface_belief[i+start_index])
    
    end_main_time = time.time()
    print(f"Total time for main function: {end_main_time - start_main_time}")
           
        

 
 
if __name__ == "__main__":
    main()