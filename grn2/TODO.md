1) Create a record of BaseModel (*5)
`python3 exp.py` (file has been reset, it is not correct now)

2) Export the data in an easy-to-exploit format
`python3 make_model_ref.py -f output/results/setup_basic/expliis_m*.stats.csv -o output/results/setup_basic/export/ref_basic.csv`

3) Create a GRN lib, using stuff from SDE
4) Create a GRN parser lib (with JSON)  

5) Create mandatory genes in GRN and external interpreter (for instance, a gene that tells when a cell must divide)

6) Integrate the GRN into a submodel  
--
7) Validate submodel in a jupyter notebook for full run
8) Create an easy interface having (1) param list, (2) running function, (3) evaluating function

9) Test PyGAD
10) Create a PyGAD workflow

11) Build a quick starting point
12) Run and debug until having another idea

Best :
Total fitness : 4.109000005554596
Best 1.9666161339887755
>> G_0: init: 9.99; noise: 5.91; b: 2.30; m: 2.41; expr: 4.90; deg: 3.38; thr: 5.94; theta: 2.95; tree : ((((2 AND 3) AND 1) OR NOT 4) AND 0)
>> G_1: init: 1.05; noise: 6.54; b: 4.26; m: 9.02; expr: 5.09; deg: 5.30; thr: 8.50; theta: 3.55; tree : (NOT 0 OR 2)
>> G_2: init: 1.31; noise: 4.28; b: 5.10; m: 3.79; expr: 3.29; deg: 7.24; thr: 5.27; theta: 6.53; tree : NOT 1
>> G_3: init: 4.62; noise: 1.49; b: 6.51; m: 5.87; expr: 6.67; deg: 7.64; thr: 4.57; theta: 1.21; tree : (0 AND 2)
>> G_4: init: 9.21; noise: 6.02; b: 6.32; m: 6.76; expr: 2.52; deg: 1.31; thr: 3.41; theta: 0.53; tree : (0 AND (2 OR 1))


Protocol :

Always create a .py file to run the experiment with a special name (ideally, the name of the experiment shall be the name of the file)
The the file must be launched with a .sh containing the date of launch in order to keep track of the schedule.

Once an experiment is run, we must move it to the "experiment" folder to clear the space
