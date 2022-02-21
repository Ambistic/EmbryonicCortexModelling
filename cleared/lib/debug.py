import pickle
import os
from pathlib import Path


def save_input(inputs, name):
    # assert repertory exists
    root = Path(os.getcwd()) / ".debug"
    root.mkdir(parents=True, exist_ok=True)
    filename = str(root / name)
    with open(filename, "wb") as f:
        pickle.dump(inputs, f)
        
    print(f"{name} inputs were saved at {filename} !")

    
def load_input(name):
    root = Path(os.getcwd()) / ".debug"
    filename = str(root / name)
    with open(filename, "rb") as f:
        return pickle.load(f)

    
def debugger(func):
    def wrapper(*args, **kwargs):
        try:
            inputs = (pickle.dumps(args), pickle.dumps(kwargs))
        except:
            inputs = None
            
        try:
            return func(*args, **kwargs)
        except:
            if inputs:
                save_input(inputs, func.__name__)
            raise
    return wrapper


def saver(func):
    def wrapper(*args, **kwargs):
        try:
            inputs = (pickle.dumps(args), pickle.dumps(kwargs))
            save_input(inputs, func.__name__)
        except:
            print("Could not save the inputs")
        return func(*args, **kwargs)
    return wrapper


def loader(func):
    def wrapper(*args, **kwargs):
        try:
            pck_args, pck_kwargs = load_input(func.__name__)
            args, kwargs = pickle.loads(pck_args), pickle.loads(pck_kwargs)
        except:
            print("Could not load the inputs")
        return func(*args, **kwargs)
    return wrapper