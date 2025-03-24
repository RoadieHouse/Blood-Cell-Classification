from fastai.vision.all import load_learner
import pathlib

# To avoid WindowsPath error when using on Linux-based systems:
temp = pathlib.PosixPath

# If you saved the model on Windows, reassign the WindowsPath to PosixPath
pathlib.WindowsPath = pathlib.PosixPath

# Load the existing model
learner = load_learner('resnet50_model.pkl')

# Re-save it with a new name in a Linux-compatible format
learner.export('resnet50_model_unix.pkl')
