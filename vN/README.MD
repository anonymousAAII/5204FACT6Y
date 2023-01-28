# Reproduced paper framework 
### Online Certification of Preference-Based Fairness for Personalized Recommender Systems
Auditing for envy in recommender systems[^1]
#### How this code works?
For this project experiments can be ran to audit for envy(-freeness) in recommender systems.
This requires for the code to generate a recommender system with recommender policies for users.
To achieve this a pipeline is build that synthesizes user data and which generates a recommender system.
The pipeline is build in a rather modular manner to save computation time.
For each step in the pipeline intermediate states are stored such that they only have to be created once and can effortlessly be retrieved.
When one likes to redo running a stage of the pipeline that's possible by **manually deleting** the created intermediate object file. 
Note however that some computations determine the computations of the subsequent stages. 
For changes in the deleted component to work through all the way through the pipeline **subsequent objects files ought to also be deleted**. 
To make carefull decisions which object files to delete taking computation time into consideration the following info is included:

The intermediate stages object files are stored in the ```/src/variables``` folder.
The subfolders ```/fm``` and ```/mv``` represent the data set which is used as a recommender system (Last.fm[^2] or MovieLens[^3] in this case). The following table shows the pipeline in sequential order, the corresponding object files and a random pick of duration of each stage:

#### Last.fm
| Pipeline stage                                | Object file/folder created          | Duration sequental (sec - min)  |  Duration multiprocessing batch[^4] |
| :---                                          |    :----:                           |    :----:                           |       ---: |
| Generating ground truth model                 | ground\_truth\_model                |-|397.39|  
| Generating ground truth                       | ground\_truth                       |0.09|| 
| Generating recommender preference est. models | recommendation\_system_est\_models  |(SVD) 5658.18|(SVD) 3331.07| 
| Generating recommendation policies            | recommendation\_policies            |||                                                   
| Generating rewards and its expectation        | rewards & expec\_rewards            ||| 
| Experiment 5.1                                |  5.1/                               ||| 

#### MovieLens
| Pipeline stage                                | Object file/folder created          | Duration sequental (sec - min)  |  Duration multiprocessing batch[^4] |
| :---                                          |    :----:                           |    :----:                           |       ---: |
| Generating ground truth model                 | ground\_truth\_model                ||| 
| Generating ground truth                       | ground\_truth                       ||| 
| Generating recommender preference est. models | recommendation\_system_est\_models  ||| 
| Generating recommendation policies            | recommendation\_policies            |||                                                   
| Generating rewards and its expectation        | rewards & expec\_rewards            ||| 
| Experiment 5.1                                | 5.1/                                ||| 

These values are an arbitraty pick from the log files which stores the wall clock execution times each time the code is ran and the pipeline stage has to be (re)build from scratch, i.e. the object file/folder does not yet exist.
```
/src/results/execution times/<data_set>.log
```

Please run the code through the terminal by going to the ```src/``` folder and running the following command:

```
# on linux
OPENBLAS_NUM_THREADS=1 python main.py
```

The program will *ask which experiment* you would like to execute. The following legenda describes the numbering: 

#### Experiments legenda
| Experiment       | Description                          | Corresponding fig. in *original* paper[^5] | 
| :-               |    :----:                            |                                       ---: |
| 5.1              | average envy / number of factors & prop.of envious users / number of factors | Figure 2                                   |


### Credits
Meta AI:
virginie.do@dauphine.eu, scd@fb.com, jamal.atif@lamsade.dauphine.fr, usunier@fb.com

[Last.fm](https://www.last.fm/)

[https://grouplens.org/](https://grouplens.org/)

[^1]: Virginie Do, Sam Corbett-Davies, Jamal Atif, Nicolas Usunier; LAMSADE, Université PSL, Université Paris Dauphine, CNRS, France
[^2]: https://grouplens.org/datasets/hetrec-2011/
[^3]: https://grouplens.org/datasets/movielens/1m/
[^4]: A batch of models is assigned to a CPU core.
[^5]: https://arxiv.org/abs/2104.14527
