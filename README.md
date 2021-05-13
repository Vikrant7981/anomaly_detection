# anomaly_detection_velc

Here we have tried to implement the research paper "VELC: A New Variational AutoEncoder Based
Model for Time Series Anomaly Detection"

Paper link :- https://arxiv.org/abs/1907.01702

Model file that is to be run is velc_model.py

# anomaly_detection_simple_vae_lstm
Here we have tried to implement the time series anomaly detection using VAE using where the encoder and decoder layers are simple forward directional LSTMs.


Model file that is to be run is simple_vae_lstm_model.py

# Data
The code uses NASA bearing data set for training and test. The bearing data has been uploaded to the folder named "dataset" here itself in the repository. 

Dataset link :- https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

We have done some pre-processing and the final output "Bearing_dataset.csv" datatset file that is used by model is present in "dataset" folder. To Show some insights about the raw data, we have generated some graphs as well.
Code that is used in pre-processing and to generate insights about data is also available in "pre_processing_insights.py".



## Code uses the rest of the folders to save model and the Images.

