## Project: Perception Pick & Place
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
[//]: # (Image References)

[image1]: ./image/voxel.png
[image2]: ./image/ransac.png
[image3]: ./image/ransac.png
[image4]:  ./image/segmentation.png
[image5]: ./image/confusion_matrix_1.png
[image6]:   ./image/confusion_matrix_2.png
[image7]:  ./image/confusion_matrix_3.png
[image8]:  ./image/world_2.png
[image9]:  ./image/world_21.png
[image10]: ./image/world_3.png
[image11]:  ./image/Normalized_confusion_matrix_1.png
[image12]:  ./image/normalized_confusion_matrix_2.png
[image13]:  ./image/normalized_confusion_matrix_3.png
---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

The goal of this project is to program a PR2 to identify and acquire target objects from a cluttered space and place them in a bin. The project is based on the Amazon's Robotics Challenge.
Objective :
1. The PR2 robot perceive its environment.
2. Identify the objects in the env 
3. Grasp and place the objects in the bin

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
The Aim of exercise is to apply some filtering techniques to generate the delibrate point cloud.
##### Step 1: VoxelGrid Downsampling Filter
   Downsampling is done to decrease the density/resolution of the pointcloud that is obtained from RGB-D camera. The voxel(volume element) is the cubic element with leaf size(sides). The point cloud is divided into voxels of specific leaf size and the spatial average of the points in the cloud is statistically combined into one output point. The increase in voxel adds more detail and processing time.

![alt text][image1]
I used LEAF_SIZE of `0.01`. Using the LEAF_SIZE less than that will increase the computations but reduce the details in object. 

##### Step 2: Passthrough Filter
   The passthrough filter is method for capturing region of interest 3D point cloud along the global cartesian axis. For processing the area in front of the PR2 Robot, I used a range of -0.4 to 0.4 in Y axis and 0.61 to 0.9 in Z axis to capture the scene in front of the PR2. 
          
##### Step 3: RANSAC Plane Fitting
   Random Sample Consensus(RANSAC) method used to identify the dataset point and classify them as inliers (model with a specified set of parameters) and outliers.

![alt text][image2]
#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  
Aim of the exercise is segmentation of point cloud data using color and spatial properties.
The exercise explain two clustering technique - 1) K-mean 2) Euclidean Clustering
1) K-mean used to cluster object based on the their distance from a randomly chosen centriod. The number of group to be clustered should be predefined. 
2) Euclidean Clustering (or DBSCAN) is density based spatial cluster of applications with noise. It groups data based on threshold distance from the their nearest neigbor.

Step 4: The conversion of XYZRGB cloud point to XYZ point cloud data to get only spatial data. Although, DBSCAN can be color data, intensity values, or other numerical features and cluster based on other features as well. To decrease the computation, K-dimensional tree is made to search the neigbors.

Step 5: The Euclidean clustering is used to build the individual point cloud for each object. The cluster-mask point cloud allows each cluster to be visualized separately.
Search the k-d tree fro clusters and extract indicies for each cluster. Assign a random
color to each isolated object in the scene. 

![alt text][image3]

To run the filtering and segmentation code:
First run the sensor_stick spawn launch, which will launch the Gazebo and Rviz.
```
$ roslaunch sensor_stick robot_spawn.launch
```
Then, to apply the filter and segmentation, Open in new terminal:
```
$ ./segmentation.py
```
In Rviz, select the `pcl_objects` under topic list to get the result

![alt text][image4]
#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Aim of the this exercise is to extract feartures from the segmented cloud. And recognize the object. 

##### Step 6:  Capture Object features : 
   The image is convert into HSV from RGB. 
   This will help us identify the object in changing lighting conditions. This is with `OpenCV`:
   ```
   hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
   ```
   
   The color histograms are used to measure object's pattern rather than just spatial data. To give model a more complete understanding the objects are positions in random orientations. The binning sceheme is used for histogram. The number of dins is propotional the detail of each object mapped. The code for histogram is in `feature.py`. The function `compute_color_histograms()` will color analysis on each point of point cloud, then add the histogram, compute features, concatenate them, and normalize them for the surface normals function `compute_normal_histograms`.
   I used *nbins size = 128*, and *bin_range = 0, 256* for color histogram and normal histogram bin range = -1, 1
   To modify how many times each object is spawned randomly, look for the for loop in `capture_features.py` that begins with `for i in range(5)`: Increase this loop to increase the number of times you capture random orientations for each object. I used 50.
   Also, To use *HSV*, find the line where `compute_color_histograms()` is called and change the flag to using_`hsv=True`.
   In capture_feature edit the model based on the world and pick_list and generate `training_set.sav` file.

##### Step 7: Train SVM model:
   `Supervised Vector Machines(SVM)` is a supervised machine learning algorithm that allows you to characterize the parameter space of your dataset into discrete classes. It is a classification technique that uses hyperplanes to delineate between discrete classes. The ideal hyperplane for each decision boundary is one that maximizes the margin, or space, from points.
   A support vector machine is used to train the model. The training_set.sav file generated is used for cross validaion of model (I used 50 fold cross validation). Run the train_svm.py script to train SVM, which will save the model in model.sav file. The train_svm.py produce the normalized confusion matrix and confusion matrix, without normalization. The confusion matrix will depict the accuracy of model. 

Scikit-Learn is used to implement the SVM algorithm
   ```
   svc = svm.SVC(kernel='linear', C = 0.01)
   ```
 The type of delineation can be changed. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
 
## **Project Perception Pipeline **

In project, we need to capture the features of the objects that we required to find from three different worlds. So instead of using the PR2 project for capturing the features(taking photographs from RGB-D camera from different angles), I used sensor_stick with few modification, such as copy models from the PR2 project to sensor stick. Copy the `pick_list.yaml` for all the worlds into config folder. Also, In `capture_feature.py` edit the model list based on the test.world and change the `training_set_#.sav` to prevent overlap.

We are now ready to **capture object features**

Open VM. Launch ROS envr :
```
roscore
```
In new terminal, Launch the `sensor_stick` Gazebo environment with 
```
$ roslaunch sensor_stick training.launch
```
Then, run the feature capture script:
```
$ rosrun sensor_stick capture_features.py
```
This will take some time running on  VM.

Once finished, you will have your features saved in the directory. I copied the file to scripts folder. We'll be using them in the next step when we train our classifier!

**Train the SVM Classifier**

now I am going to train our classifier on the features I've extracted from our objects. `train_svm.py` is going to do this.
```
$ cd ~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/
```
Open `train_svm.py` and make sure line ~39 or so is loading the correct feature data we just produced:
```
training_set = pickle.load(open('training_set_#.sav', 'rb'))
```
Then, we're going to check our SVC kernel is set to linear:
```
clf = svm.SVC(kernel='linear', C = 0.01)
```
Now, from this same directory, run the script:
```
$ rosrunh sensor_stick train_svm.py
```
Two confusion matrices will be generated--one will be a normalized version. 

**World 1**

![alt text][image5]
![alt text][image11]
Accuracy: 0.98 (+/- 0.17)

**World 2**
![alt text][image6]
![alt text][image12]
Accuracy: 0.96 (+/- 0.17)

**World 3**
![alt text][image7]
![alt text][image13]
Accuracy: 0.94 (+/- 0.17)


**Label in RViz**

Now, we can check the labeling of objects in RViz. Exit out of any existing Gazebo or RViz sessions, and type in a terminal:
```
$ roslaunch pr2_robot pick_place_project.launch
```
Once it is loaded. Run in new terminal:
```
rosrun pr2_robot project_template.py
```
Save the `model_#.sav` in same directory as `project_template.py`.
![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

For each test#.world These are the images.
![alt text][image8]

![alt text][image9]

![alt text][image10]
### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

And here's another image! 
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

I did pick and place code but due to time constraint I didn't add the results here. 

