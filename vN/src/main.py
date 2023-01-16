import pandas as pd
import numpy as np
from scipy import sparse
import implicit

def merge_duplicates(df, col_duplicate, col_merge_value, mode="sum"):
    """
    merge_duplicate checks for duplicate entries 
    and according to the given mode performs an operation on the specified column value.

    :df: dataframe used
    :col_duplicate: name of column to be checked for duplicates
    :col_merge_value: name of column which value to perform an operation on when it concerns a duplicate
    :mode: name which determines which operation is performed, default is 'sum'
    :return: dataframe which contains the unique entries and their values after the operation is performed
    """ 
    if mode == "sum":
        return df.groupby(col_duplicate, as_index = False)[col_merge_value].sum()
    # Default sum the <col_merge_value> values of the duplicates
    else:
        return df.groupby(col_duplicate, as_index = False)[col_merge_value].sum()

if __name__ == "__main__":
    # Source folder of dataset
    DATA_SRC = "../data/hetrec2011-lastfm-2k/"
    # Mapping from data variable name to its filename
    data_map = {"artists": "artists.dat", 
                "tags": "tags.dat", 
                "user_artists": "user_artists.dat",
                "user_taggedartists": "user_taggedartists.dat",
                "user_taggedartists-timestamps": "user_taggedartists-timestamps.dat",
                "user_friends": "user_friends.dat"}

    # Data structs for datasets for computation (dictionaries have fast access O(1))
    artists_ds = {} # {<id>: {"name": <name>, "url": <url>}}
    tags_ds = {} # {<tagID>: <tagValue>}
    user_artists_ds = {} # {<userID>: {<artistID>: <weight>}}

    var_names = list(data_map.keys())
    
    # Global accesible variables
    myVars = vars()
    
    # Read in data files
    for var_name, file_name in data_map.items():
        # Read data in Pandas Dataframe (pure for manual exploration and visualization)    
        myVars[var_name] = pd.read_csv(DATA_SRC + file_name, sep="\t",  encoding="latin-1")

        # Open file, read data and save in appropriate format for fast computation
        with open(DATA_SRC + file_name, mode="r", encoding="latin-1") as f:
            # Column names
            first_line = f.readline()
            column_names = first_line.strip().split("\t")
            # print(column_names)
    
            for line in f:
                # print(line)
                
                # artists.dat
                # id \t name \t url \t pictureURL
                # 707	Metallica	http://www.last.fm/music/Metallica	http://userserve-ak.last.fm/serve/252/7560709.jpg
                if var_name == var_names[0]:
                    entry = line.strip().split("\t")
                    # {<id>: <name>}
                    artists_ds[int(entry[0])] = entry[1]

                # tags.dat
                # tagID \t tagValue
                # 1	metal
                elif var_name == var_names[1]:
                    entry = line.strip().split("\t")
                    # {<tagID>: <tagValue>}
                    tags_ds[int(entry[0])] = entry[1]

                # user_artists.dat
                # userID \t artistID \t weight
                # 2	51	13883                    
                elif var_name == var_names[2]:
                    entry = np.array(line.strip().split("\t"), dtype=int)
                    # {<userID>: {<artistID>: <weight>}}
                    # User exists
                    if int(entry[0]) in user_artists_ds:
                        # Artist doesn't exist
                        if not (int(entry[1]) in user_artists_ds[int(entry[0])]):
                            user_artists_ds[int(entry[0])][int(entry[1])] = int(entry[2])
                    # Create first user entry
                    else:
                        user_artists_ds[int(entry[0])] = {int(entry[1]): int(entry[2])}
                
                # user_taggedartists.dat
                # userID \t artistID \t tagID \t day \t month \t year
                # 2	52	13	1	4	2009  
                elif var_name == var_names[3]:
                    break
                    # OPTIONAL 
                    entry = np.array(line.strip().split("\t"), dtype=int)
                                    
                # user_taggedartists-timestamps.dat
                # userID \t artistID \t tagID \t timestamp
                # 2	52	13	1238536800000    
                elif var_name == var_names[4]:
                    break
                    # OPTIONAL 
                    entry = np.array(line.strip().split("\t"), dtype=int)
                
                # user_friends.dat
                # userID \t friendID
                # 2	275    
                elif var_name == var_names[5]:
                    break
                    # OPTIONAL 
                    entry = np.array(line.strip().split("\t"), dtype=int)
                else:
                    raise Exception("Something goes wrong reading the file")                    

            f.close()

    # print(user_artists_ds)
    # print(user_artists_ds[2][51])
    # print(user_artists)

    # Get user and artists ids
    users = np.array(user_artists["userID"].unique())
    artists = np.array(user_artists["artistID"].unique())

    print("artists =", artists)
    print("users =", users)

    # Mapping from user_id to index in users and sparse-matrix (R) and vice versa
    users_index = dict(zip(users, np.arange(len(users))))
    index_users = dict(zip(np.arange(len(users)), users))
    # Mapping from artists_id to index in artists and sparse-matrix (R) and vice versa
    artists_index = dict(zip(artists, np.arange(len(artists))))
    index_artists = dict(zip(np.arange(len(artists)), artists))

    # User-item observation matrix (Johnson 2014)    
    R = np.zeros((len(users), len(artists)))

    # interaction = user_artists[((user_artists["userID"] == 2) & (user_artists["artistID"] == 51))]
    # print(interaction)

    # Initialize user-item observation matrix    
    for index, user in enumerate(users):
        for column, artist in enumerate(artists):

            # When user_artist does not exist
            if not (artist in user_artists_ds[user]):
                continue
            
            # Save user-item (user_artist) interaction
            R[index][column] = user_artists_ds[user][artist]

    print("User-item observation matrix", "\n", R)
    user_item_matrix = sparse.csr_matrix(R) 

    # initialize a model
    model = implicit.als.AlternatingLeastSquares(factors=50)

    # train the model on a sparse matrix of user/item/confidence weights
    model.fit(user_item_matrix)

    # recommend items for a user
    recommendations = model.recommend(users_index[2], user_item_matrix[users_index[2]])    

    print(recommendations)

    for item_index in recommendations[0]:
        print(index_artists[item_index])

    # artist_streams = merge_duplicates(user_artists, "artistID", "weight")
    # rank_artist_streams = artist_streams.sort_values(by=["weight"])
    # print(rank_artist_streams)
    
    