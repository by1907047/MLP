# Flexible Calorimetric Flow Sensor with Unprecedented Sensitivity and Directional Resolution for Multiple Flight Parameter Detection
## AOA and AOS estimation
For AOA and AOS estimation, a series of MLP neural networks were trained to estimate the AOA and AOS values of the airfoil using data collected from the FCF sensor during wind tunnel testing to evaluate the estimation accuracy for each input pattern (PV1, PV2, PV3, and PV1-3). The inputs of each network were the normalized values of the corresponding signal channels, and the outputs were AOA and AOS values. To avoid estimation errors caused by different network structures, the same learning rate, activation function, hidden layer, and number of nodes were set for each network.
## Relative airflow velocity estimation
In the flight velocity estimation experiment, eight channels of data from the P1-2 and V1-2 outputs of two FCF sensors were used as inputs to train the MLP neural network to estimate the flight velocity of the MAV and the outputs were the three velocity components in the body coordinate system. After training and optimization, the network model was set up with three hidden layers and the number of nodes in each layer was 100. Training was performed by combining both indoor and outdoor test flight data. All data from the slide reciprocating motion, indoor flight, and outdoor flight tests were combined in the training set to improve the estimation performance of the MLP because the velocity ranges and testing conditions (e.g., propeller spinning and atmospheric conditions) between datasets were different. Estimation performance was validated using indoor and outdoor flight datasets collected independently of those used for training.
## Code
MLP_AOA_AOS.py is the training and testing code for the MLP model used for AOA and AOS estimation.
MLP_velocity.py is the training and testing code for the MLP model for relative airflow velocity estimation.
## Dataset
The AOA_AOS_10_ms-1 folder contains the training and testing sets at 10 ms<sup>-1</sup> flow velocity.  
The AOA_AOS_30_ms-1 folder contains the training and testing sets at 30 ms^-1^ flow velocity.  
The Velocity_indoor folder contains the training and testing sets during indoor flight.  
The Velocity_outdoor folder contains the training and testing sets during outdoor flight.  
The training set is the same for both indoor and outdoor flights.
## Model
The AOA_AOS_10_ms-1 folder includes the trained MLP models for the four input patterns at 10 ms^-1^ flow velocity.
The AOA_AOS_30_ms-1 folder includes the trained MLP models for the four input patterns at 30 ms^-1^ flow velocity.
The Velocity_indoor folder includes the trained MLP model for indoor flight.
The Velocity_outdoor folder includes the trained MLP model for outdoor flight.
## Citation
If you find the code helpful, please cite the paper:
