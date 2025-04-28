import os

from create_bd import create_bd


def load_bd():
    if os.path.exists("data.db"):
        return
    else:
        create_bd("reviews.csv")
