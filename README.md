# ATNLP-individual
## Getting code and data 
Create a directory for code and dataset, for example: \
`mkdir ATNLP` \
Download data and code in this folder \
`cd ATNLP`\
`git clone https://github.com/Adrianna40/ATNLP-individual.git`\
`git clone https://github.com/brendenlake/SCAN.git`\
Go to directory with code\
`cd ATNLP-individual`
## Environement 
`pip install -r requirements.txt`
## Experiment 1b 
To reproduce results with a use of T5 model, run:\
`python3 exp1b.py <data_size> t5`\
where <data_size> can have value 1, 2, 4, 8, 16, 32 or 64. \
To reproduce results with a use of CodeT5 model, run:\
`python3 exp1b.py <data_size>`
## Experiment 2 
To reproduce results with a use of T5 model, run:\
`python3 exp2.py t5`\
To reproduce results with a use of CodeT5 model, run:\
`python3 exp2.py`
