import os
import shutil
import time
from scripts.preprocess import resample, slice
from scripts.predict import predict
from scripts.recreate import recreate, transform
from scripts.ensemble import ensemble
from scripts.aggregate import aggregate
from scripts.calculate import calculate

start_time = time.time()

# Remove previous run data
for folder in ['preprocessed', 'predicted', 'output']:
    if os.path.exists(folder):
        shutil.rmtree(folder)

print("----------------------------------------")
print("Preprocessing: \n")
resample()
slice()
print("----------------------------------------")
print("Running Models: \n")
predict()
print("----------------------------------------")
print("Creating Prediction: \n")
recreate()
transform()
ensemble()
aggregate()
print("----------------------------------------")
print("Calculating Lesion Volume and Loads: \n")
calculate()

end_time = time.time()
total_runtime = end_time - start_time
print("----------------------------------------")
print("LesSeg Complete! --> Please check 'output' folder")
print(f"Total runtime: {total_runtime} seconds\n")

