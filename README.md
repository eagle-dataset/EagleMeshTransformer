# EAGLE: Large-scale Learning of Turbulent Fluid Dynamics with Mesh Transformers
 
This repository contains the code associated to the paper <a href="https://openreview.net/forum?id=mfIX4QpsARJ">EAGLE: Large-scale Learning of Turbulent Fluid Dynamics with Mesh Transformers</a>

Link: <a href="https://eagle-dataset.github.io/"> Project page </a>

# Abstract
Estimating fluid dynamics is classically done through the simulation and integration of numerical models solving the Navier-Stokes equations, which is computationally complex and time-consuming even on high-end hardware. This is a notoriously hard problem to solve, which has recently been addressed with machine learning, in particular graph neural networks (GNN) and variants trained and evaluated on datasets of static objects in static scenes with fixed geometry. We attempt to go beyond existing work in complexity and introduce a new model, method and benchmark. We propose EAGLE, a large-scale dataset of ∼1.1 million 2D meshes resulting from simulations of unsteady fluid dynamics caused by a moving flow source interacting with nonlinear scene structure, comprised of 600 different scenes of three different types. To perform future forecasting of pressure and velocity on the challenging EAGLE dataset, we introduce a new mesh transformer. It leverages node clustering, graph pooling and global attention to learn long-range dependencies between spatially distant data points without needing a large number of iterations, as existing GNN methods do. We show that our transformer outperforms state-of-the-art performance on, both, existing synthetic and real datasets and on EAGLE. Finally, we highlight that our approach learns to attend to airflow, integrating complex information in a single iteration.



# Dataset
You can download the dataset on this <a href="https://datasets.liris.cnrs.fr/eagle-version1"> link</a>. Simulations data are stored in a single numpy archive file (.npz), containing nodes 2D positions, types, velocity and pressure. The edges are stored in another file (triangles.npy) as triplet of points. Below are the SHA25 sums of each files:
```

f1bbc1dc22b0fbc57a5f8d0243d85f6471c43585fb0ecc7409de19996d3de12c  eagle_clusters.tar.gz
f73cb9a443011646fb944e0a634a0d91c20b3d71a8b4d89d55486f9e99bdca78  spline.tar.gz
ac04d3efb539a80d8538fb8214228652b482ab149fc7cc9ecf0b6d119e3b1be7  step.tar.gz
59a2ae96ca5ade7d3772e58b302c4132e1ee003ac239b7e38973ceb480a979e6  triangular.tar.gz
```
 # Training
 The mesh-transformer module can be trained using the corresponding script ```train_graphvit.py```. Below is a description of the relevant parameters :
 
```
     --epoch          : Number of epoch for training. Evaluate the trained model if set to 0
     --lr             : Learning rate
     --dataset_path   : Location of the dataset
     --alpha          : Weight on the pressure term in the loss
     --horizon_val    : Number of timestep to validate on
     --horizon_train  : Number of timestep to train on
     --n_cluster      : Number of nodes per clusters
     --w_size         : Dimension of the latent representation of a cluster
     --batchsize      : ...
     --name           : Name for saving/loading weights
     
```
Note that the script using the dataloaders in ```Dataloaders```. You will have to specify the location of the dataset. We strongly recommend to pre-compute the clusters for each simulations in the dataset using the following script:
``` python3 clusterize_eagle.py --max_cluster_size XXX --geom "Spl" --traj 1```

Clustering script is configured with the geometry type "Cre, Tri, Spl" and the direction of flight (1 or 2). This allows to run the clustering in parallel easily.

# Citation
```    @inproceedings{janny2023eagle,
        title = "EAGLE: Large-scale Learning of Turbulent Fluid Dynamics with Mesh Transformers",
        author = {Steeven Janny and
                  Aurélien Benetteau and
                  Nicolas Thome and Madiha Nadri and Julie Digne and Christian Wolf},
        booktitle = "International Conference on Learning Representations (ICLR)",
        year = "2023"}
```
