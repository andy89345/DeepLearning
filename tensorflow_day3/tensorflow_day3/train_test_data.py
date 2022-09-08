import pandas as pd
class train_test_data:
    train = pd.read_csv(
    "train.csv",
    names=[ 
            "survived",
            "sex",
            "age",
            "n_siblings_spouses",
            "Shucked weight",
            "parch",
            "fare", 
            "class",
            "deck",
            "embark_town",
            "alone",
            ]
    )

    test = pd.read_csv(
    "eval.csv",
    names=[ 
            "survived",
            "sex",
            "age",
            "n_siblings_spouses",
            "Shucked weight",
            "parch",
            "fare", 
            "class",
            "deck",
            "embark_town",
            "alone",
            ]
    )
    


