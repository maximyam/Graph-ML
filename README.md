# Graph-ML


## File structure 
data/ - contains data (not pushed into repo)
markdown/ - 
-> project_goals.md => goal of project
-> data_processing_walkthrough.md => explain 01-data_processing.ipynb
src/ - contains model, pipeline, additional tools 
temp/ - throw all things you dont want to push here 
01-data_processing.ipynb -> Step 1


!!! After running 01-data_processing.ipynb, you may use all the new processed data for work  !!!



--- How teammates reload (no need to rerun the pipeline) ---
import anndata as ad
work = ad.read_h5ad("data/processed/work_processed.h5ad")   # full object (has obsm["X_features"], uns["feature_blocks"])
import numpy as np, scipy.sparse as sp, pandas as pd, json
X    = np.load("data/processed/X_features.npy")             # augmented node features
cols = json.load(open("data/processed/feature_blocks.json"))# {"pca":[0,30], "markers":[...], ...}
A    = sp.load_npz("data/processed/connectivities.npz")     # kNN adjacency (sparse)
meta = pd.read_csv("data/processed/cell_metadata.csv", index_col=0)  # labels + masks + local_homophily


## How to run 
1. activate environment from yml file.  or you may use gnn_course environment (conda activate gnn_course)
2. pip install requirements (we will fix the yml file later) # pip install -r requirements.txt
3. Run 01-data_processing.ipynb (you will create data folder and download there)

Run `01-data_processing.ipynb` first, then `02-graph_engineering_baselines.ipynb`.
Later runs can reuse processed data and matching graph exports. Baselines train again.
Graph outputs are saved under `data/graph_engineering/`. See the
[walkthrough](markdown/graph_engineering_walkthrough.md) for details.
The older `environment_osx_intel.yml` is the original course environment, not the
setup used for these notebooks.

dont touch .gitignore unless youre adding more stuff to ignore (env files etc..)
avoid pushing large data into repo





## tldr on git 
everyone can push and pull files. lets all work on main but update the group when we push something new! pls ensure it works / dont break pipeline 

### to retrieve latest 
git pull 

### to push update 
git add .  / git add (type files manually)
git commit -m "[NAME] whatyoudid" 
git push 


## Last update
Maxim pushed testing code 
