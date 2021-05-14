# anomaly detection usiing VELC: A New Variational AutoEncoder Based Model

We have tried implemented the research paper "VELC: A New Variational AutoEncoder Based
Model for Time Series Anomaly Detection"

Paper link :- https://arxiv.org/abs/1907.01702
Model file :- velc.py


# anomaly detection using vae lstm with a re-encoder
Along with the VELC model we have also implement the time series anomaly detection using VAE where the encoder, decoder and a re-encoder layers, which are  Bi-directional LSTMs.
Model :- vae_with_ReEncoder.py


# anomaly detection using simple vae lstm
Along with the VELC model we have also implement the time series anomaly detection using VAE where the encoder and a decoder layers, which are  Bi-directional LSTMs.
Model :- simple_vae_lstm_model.py



# Data
The code uses NASA bearing data set for training and test. The bearing data has been uploaded to the folder named "dataset" here itself in the repository. 

Dataset link :- https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

We have done some pre-processing and the final output "Bearing_dataset.csv" datatset file that is used by model is present in "dataset" folder. To Show some insights about the raw data, we have generated some graphs as well.
Code that is used in pre-processing and to generate insights about data is also available in "pre_processing_insights.py".

# References
https://github.com/shaohua0116/VAE-Tensorflow

https://towardsdatascience.com/machine-learning-for-anomaly-detection-and-condition-monitoring-d4614e7de770

https://towardsdatascience.com/how-to-use-machine-learning-for-anomaly-detection-and-condition-monitoring-6742f82900d7

https://towardsdatascience.com/variational-autoencoders-as-generative-models-with-keras-e0c79415a7eb



## Code uses the rest of the folders to save model and the Images.

