# ECG
code workflow:
(a) data prepration
input: raw ECG sequences
preprocessing steps:
(1) extract R peaks 
(2) remove baseline from raw signals (see lines 210-222)
(3) extract sliding window from the filtered ECG signals around each R peak, the observations are {90 past frames,  target frame, 89 frames}
output: a 'dict' python object stored in pickle file format
'dict' layout:
key--'samples': a list object, each item is a dict of ecg channels: {'MLII', 'V1'}, each channel is a list with 180 real floats which are 
key--'labels': a list object, each item is a single integer which is the class id
key--'sindrs': a list object, each item is a single integer refers to the number of samples in each record
key--'rinds': a list object, each item is a single integer indicates the frame index of a detected beat segment in its record
key--'rlocs': a list object, each item is a single integer indicates the frame index of a groundtruth beat segment in its record

(b) dataset prepration
split the data into training, validation, testing sets from the whole dataset with proportion of 0.5, 0.15, 0.35
save them into individual pickle files with same as (a.3)

(c) feature selection&extraction
(1) call function 'preparedataset' in ecg_dataset.py for each saved pickle file in (b)
(2) reconstruct the input observations for GRU model as done in ecg_leads.py
(3) call function 'run' in console.py for each training configurations as done in run.py, runcnn.py, rungru.py, runsvm.py
