1) Create a record of BaseModel (*5)
`python3 exp.py`

2) Export the data in an easy-to-exploit format
`python3 make_model_ref.py -f output/results/setup_basic/expliis_m*.stats.csv -o output/results/setup_basic/export/ref_basic.csv`

3) Create a GRN lib, using stuff from SDE
4) Create a GRN parser lib (with JSON)
5) Create mandatory genes in GRN and external interpreter (for instance, a gene that tells when a cell must divide)

6) Integrate the GRN into a submodel
7) Create an easy interface having (1) param list, (2) running function, (3) evaluating function

8) Test PyGAD
9) Create a PyGAD workflow

10) Build a quick starting point
11) Run and debug until having another idea
