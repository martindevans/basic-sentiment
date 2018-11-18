## Datasets
 
 - `Gutenbergdammit` dataset is from <https://github.com/aparrish/gutenberg-dammit>
 - `Wikipedia` dataset is from polyglot project <https://sites.google.com/site/rmyeid/projects/polyglot>

## Running

Run `python main.py`
 - Edit tensorflow config in main.py
 - Edit tensorboard config in main.py
   - Run `tensorboard --logdir logs` and visit `localhost:6006`
 - Comment out datasets used in `w2v/main.py`
 - Change hyperparameters at the top of `w2v/main.py`