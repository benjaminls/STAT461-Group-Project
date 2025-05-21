# STAT461-Group-Project

## Setup

### Virtual Environment

Placeholder

### PowerPoint Slides and PyG Data Loaders
https://drive.google.com/drive/folders/1cN6LXsStMcSE5RdHkAYo9j68MNOem1r0?usp=sharing

### Downloading Datasets

[TrackML Kaggle page](https://www.kaggle.com/competitions/trackml-particle-identification/data)

1) Sign up for a Kaggle account

2) Navigate to the [TrackML Particle Tracking Challenge home page](https://www.kaggle.com/competitions/trackml-particle-identification/overview) and agree to competition terms and conditions.

3) Install Kaggle command line tool

4) Download dataset using Kaggle CLI

```bash
kaggle competitions download -c trackml-particle-identification
```

## Data

The data are stored as a collection of CSV files. The top-level unit of data is an "event" that contains all the information from a single collision between proton bunches. Each event is considered to be [independent and identically distributed](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) (IID) from the others. Each event contains four types of csv data files:
1) `event#-hits.csv`: Contains the raw hit data from the detector. Each hit is a 3D point in space, with an associated volume, layer, and module ID that identifies the detector element that registered the hit. Every hit is indexed by an integer unique to the event. 
2) `event#-truth.csv`: Contains the truth information for each hit. The truth information is indexed by the same integer as the hit data, allowing us to directly compare `hit_id` in the two files to match the hit data with the truth information. The truth information includes the particle ID, the 3D position and momentum of the particle at the time of the hit, and the weight of the hit. Particle ID is a unique identifier for the track that produced that hit. A collection of hits created by the same particle forms a track and every track has unique particle id in that particular event. The weight is a measure of how much the hit contributes to the overall event used for scoring the model. 
3) `event#-cells.csv`: Contains the cell data for each hit. In the detector, cells are the smallest unit of active detector material, similar to pixels in a camera, and may be useful to refine the hit-to-track association. Cells may be grouped together to form larger structures, such as layers or modules, and may have different shapes and sizes. It is useful to think about the size of the cell as being proportional to the uncertainty on the hit position. Every cell is identified by `ch0` and `ch1` values that are analogous to the x and y coordinates of the detector module. The `value` column is proportional to the amount of charge deposited in the cell by the hit. The `hit_id` column is used to match the cell data with the hit data.
4) `event#-particles.csv`: Contains the particle data for each event. It is "truth" level information in that it contains true information not available when real physics data is collected by the detector (remember, this is simulated data, so we have access to the "truth"). The particle data includes the particle ID, the initial 3D position (`vx`, `vy`, `vz`) and momentum (`px`, `py`, `pz`), the `q` charge of the particle, and the number of hits `nhits` associated with that particle. The `particle_id` column is used to match the particle data with the truth data. 


### Hits
```csv
hit_id,x,y,z,volume_id,layer_id,module_id
1,-64.4099,-7.1637,-1502.5,7,2,1
2,-55.3361,0.635342,-1502.5,7,2,1
3,-83.8305,-1.14301,-1502.5,7,2,1
```
**Columns:** hit_id,x,y,z,volume_id,layer_id,module_id

- `x`, `y`, `z`: Hit position in mm


### Truth
```csv
hit_id,particle_id,tx,ty,tz,tpx,tpy,tpz,weight
1,0,-64.4116,-7.16412,-1502.5,250710,-149908,-956385,     0
2,22525763437723648,-55.3385,0.630805,-1502.5,-0.570605,0.0283904,-15.4922,9.86408e-06
3,0,-83.828,-1.14558,-1502.5,626295,-169767,-760877,     0
```
**Columns:** hit_id,particle_id,tx,ty,tz,tpx,tpy,tpz,weight

- `tpx`, `tpy`, `tpz`: Truth momentum in GeV
- `tx`, `ty`, `tz`: Truth position in mm


### Cells
```csv
hit_id,ch0,ch1,value
1,209,617,0.0138317
1,210,617,0.0798866
1,209,618,0.211723
```
**Columns:** hit_id,ch0,ch1,value


### Particles
```csv
particle_id,vx,vy,vz,px,py,pz,q,nhits
4503668346847232,-0.00928816,0.00986098,-0.0778789,-0.0552689,0.323272,-0.203492,-1,8
4503737066323968,-0.00928816,0.00986098,-0.0778789,-0.948125,0.470892,2.01006,1,11
4503805785800704,-0.00928816,0.00986098,-0.0778789,-0.886484,0.105749,0.683881,-1,0
```
**Columns:** particle_id,vx,vy,vz,px,py,pz,q,nhits

- `vx`, `vy`, `vz`: Initial position or vertex in mm in global coordinates 
- `px`, `py`, `pz`: Initial momentum in GeV along each global axis
- `q`: Charge of the particle (units of electron charge)






## Directory Structure
- `src`: Main code goes in here
- `scripts`: Any small, standalone scripts go in here
- `utils`: Files containing common functions or objects used across the codebase
- `notebooks`: Any jupyter notebooks
- `data`: Datasets for training
- `configs`: Configuration files, if needed
- `tests`: Code containing test cases (`unittest`)
