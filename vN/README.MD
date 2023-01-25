# Reproduced: Online Certification of Preference-Based Fairness for Personalized Recommender Systems
### Auditing for envy in recommender systems

#### How this code works?
For this project experiments can be ran to audit for envy(-freeness) in recommender systems.
This requires for the code to generate a recommender system with recommender policies for users.
To achieve this a pipeline is build that synthesizes user data and which generates a recommender system.
The pipeline is build in a rather modular manner to save computation time.
For each step in the pipeline intermediate states are stored such that they only have to be created once.
When one likes to redo running a stage of the pipeline that's possible by == manually deleting == the created intermediate object file. Note however that some computations determine the computations of the subsequent stages. For changes in the deleted component to work through all the way through the pipeline ==subsequent objects files ought to also be deleted==. To make carefull decisions which object files to delete taking computation time into consideration the following info is included:

The intermediate stages are stored in the ```/src/variables``` folder.
The subfolders ```/fm``` and ```/mv``` represent the data set which is used as a recommder system (Last.fm or MovieLens in this case). The following table shows the pipeline and corresponding object files:

| Stage                                   | Object file/folder created          | Average duration (s - min)  |
| :---                                    |    :----:                           |                        ---: |
| Generating ground truth model           | ground\_truth\_model                | 480 - 8                     |
| Generating ground truth                 | ground\_truth                       | 0.07 - x                    |
| Generating rec. preference est. models  | recommendation\_system_est\_models  | 1350 - 22                   |
| Generating expectation of the rewards   | expec\_rewards                      | 28 - 0.5                    |
| Experiment 5.1                          |  5.1/                               | 4.5 - x                     |
--------------------------------------------------------------------
Please run the code through the terminal with the following command:

```
# on linux
OPENBLAS_NUM_THREADS=1 python main.py
```
--------------------------------------------------------------------

### Credits
Virginie Do, Sam Corbett-Davies, Jamal Atif, Nicolas Usunier
LAMSADE, Université PSL, Université Paris Dauphine, CNRS, France

Meta AI
virginie.do@dauphine.eu, scd@fb.com, jamal.atif@lamsade.dauphine.fr, usunier@fb.com
