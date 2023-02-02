# Reproduced paper framework 
### Online Certification of Preference-Based Fairness for Personalized Recommender Systems
*Auditing for envy in recommender systems*. This framework is build based on reproducing the paper: "Online Certification of Preference-Based Fairness for Personalized Recommender Systems."[^1][^5]. With this framework we aimed to construct a reasonable environment in which the paper could be reproduced in a controlled and structured experimental setting. For this we took into consideration runtime performance, reusability and scalibilty. For now the framework has only been applied to the two data sets Last.fm[^2] or MovieLens[^3], for a set of 3 models (ALS[^8], LMF[^7], Funky SVD[^6]) and a selection of performance metrics (precision@k, ndcg@k, dcg@k), however, it can be easily be extended to run for other data sets on which we wish to perform to reproduce the claims of the original paper.  

#### How this code works?
For this project experiments can be ran to audit for envy(-freeness) in recommender systems.
This requires for the code to generate a recommender system with its recommendation policies for users.
To achieve this a pipeline is build that completes sparse read in user data to synthesize true user preferences which we use to generate a recommender system. Based on this ground truth we can then audit the recommender system for fairness using the definition of envy-freeness as given by the original paper[^1]. The experiments implemented can provide a novel formal analysis of envy-free recommender systems.  

The pipeline is build in a rather modular manner to save computation time.
For each step in the pipeline intermediate states are stored such that they only have to be created once and can effortlessly be retrieved. The pipeline modular objects are stored as:

```
src/variables/<data_set_label>/ground_truth/<model_type>/ground_truth_model
src/variables/<data_set_label>/ground_truth/<model_type>/ground_truth 
```
Where ```<data_set_label>``` is the data set used to model .e.g. *fm* for the Last.fm dataset. ```<model_type>``` the model type used e.g. *als* for AlternatingLeastSquares. So you get:

```
src/variables/fm/ground_truth/als/ground_truth_model
```

When one likes to redo running a stage of the pipeline that's possible by **manually deleting** the created intermediate object file. 
Note however that some computations determine the computations of the subsequent stages. 
For changes in the deleted component to work through all the way through the pipeline **subsequent objects files ought to also be deleted**. 
To make carefull decisions which object files to delete taking computation time into consideration the following info is included:

The intermediate stages object files are stored in the ```/src/variables/...``` path.
The infix subfolders ```/fm``` and ```/mv``` represent the data set which is used as a recommender system (Last.fm[^2] or MovieLens[^3] in this case). And the infix ```<model_type>``` specifies which sorf of model is applied. The following table shows the pipeline in sequential order, a random pick of duration of each stage and whether a corresponding object file is created:

#### Last.fm
| Pipeline stage                                | Object file created          | Duration sequental (sec - min)  |  Duration multiprocessing batch[^4] |
| :---                                          |    :----:                           |    :----:                           |       ---: |
| Generating ground truth model                 | ground\_truth\_model                |-|385.29|  
| Predicting ground truth                       | ground\_truth                       |0.03|-| 
| Generating 9 recommender models                 | recommender\_model                  |-|364.428 (ALS) <br> 2348.44 (Funky SVD)|
| Initializing 9 recommenders                   | models/<data_set>/*|-|106.27| 
| Experiment 5.1                                |  x                               |4.42|-|

#### MovieLens
| Pipeline stage                                | Object file created          | Duration sequental (sec - min)  |  Duration multiprocessing batch[^4] |
| :---                                          |    :----:                           |    :----:                           |       ---: |
| Generating ground truth model                 | ground\_truth\_model                |-|384.16| 
| Predicting ground truth                       | ground\_truth                       |0.08|-| 
| Generating 9 recommender models | recommender\_model |-|345.72 (ALS) <br> 2296.97 (Funky SVD| 
| Initializing 9 recommenders                   | models/<data_set>/*|-|106.27| 
| Experiment 5.1                                | x                                |5.03|-| 

These values are an arbitraty pick from the log files which store the wall clock execution times each time the code is ran and the pipeline stage has to be (re)build from scratch, i.e. the object file/folder does not yet exist.
```
/src/results/execution times/<data_set>.log
```
In these files the chosen models are also logged as:
```
{"mv": 
  {"ground_truth": 
    {"ALGORITHM": "als", "METRIC": "precision"}, 
   "recommender": 
    {"ALGORITHM": "als", "METRIC": "precision", "normalize": false}
  }, 
"fm": {"ground_truth": 
        {"ALGORITHM": "als", "METRIC": "precision"}, 
    "recommender": 
      {"ALGORITHM": "als", "METRIC": "precision", "normalize": false}
   }
 }
```
Note however that this only involves when the models are constructed not when they are loaded in.

#### ! Please run the code on Linux to prevent portability issues regarding code libaries !

Please run the code through the terminal by going to the ```src/``` folder and running the following command:

```
# on linux
OPENBLAS_NUM_THREADS=1 python main.py
```

The program will *ask which experiment* you would like to execute. The legenda below describes the numbering. However, if you are not a fan of **CLI** usage a quick hack would be to use the code in *debug mode* the following way and setting the values you desire.
```
# File location: src/constant.py
DEBUG = True
DUMMY_DATA_SET_CHOICE = "all"
DUMMY_MODEL_CHOICE = {"ground_truth": {"ALGORITHM": "als", "METRIC": "precision"}, 
                        "recommender": {"ALGORITHM": "als", "METRIC": "precision", "normalize": False}} 
DUMMY_EXPERIMENT_CHOICE = "5.1"
```
#### Experiments legenda
| Experiment       | Description                          | Corresponding fig. in *original* paper[^5] | 
| :-               |    :----:                            |                                       ---: |
| 5.1              | **Sources of envy**: average envy / number of factors & prop.of envious users / number of factors | Figure 2                                   |

#### Implemented possible models per pipe line stage
|  Data set    | Ground truth model | Recommender model | 
|  :-           |    :----:       |              ---: |
| LastFM        |  (ALS) <br> (LMF)  | (ALS) <br> Funky SVD |
| MovieLens     |  (ALS) <br> (LMF)  | (ALS) <br> Funky SVD |

Since all models are saved as objects in a file these models can be reused at any later time. Data storage design however is not optimal and is open to improvement.

### Credits
Meta AI:
virginie.do@dauphine.eu, scd@fb.com, jamal.atif@lamsade.dauphine.fr, usunier@fb.com

[Last.fm](https://www.last.fm/)

[https://grouplens.org/](https://grouplens.org/)

[^1]: Virginie Do, Sam Corbett-Davies, Jamal Atif, Nicolas Usunier; LAMSADE, Université PSL, Université Paris Dauphine, CNRS, France
[^2]: https://grouplens.org/datasets/hetrec-2011/
[^3]: https://grouplens.org/datasets/movielens/1m/
[^4]: A random batch of models is assigned to a CPU core.
[^5]: https://arxiv.org/abs/2104.14527
[^6]: https://github.com/gbolmier/funk-svd
[^7]: https://benfred.github.io/implicit/api/models/cpu/lmf.html#
[^8]: https://benfred.github.io/implicit/api/models/cpu/als.html
