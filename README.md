# Norfolk_Groundwater_Model
This repository contains scripts to model and forecast the groundwater table level in Norfolk, Virginia using Long Short-term Memory and Recurrent neural networks. These models are created with Tensorflow and Keras and run on a HPC with a GPU. The models are trained and tested with observed data; the models are also tested on forecast data to simulate a real-time prediction scenario.

# Project Motivation
There is a need for accurate forecasts of groundwater table as part of flood prediction in coastal urban areas because:

- Coastal urban areas face recurrent flooding from storm events and sea level rise
  - Expected to get worse as climate change continues
- In these areas, the groundwater level is often close to the surface
  - Exact height is only known at sparse points (wells)
  - Can quickly rise in response to storms
- High groundwater level decreases storage capacity and
  - Increases runoff
  - Increases stormwater load
  - Increases flooding during storms

# Workflow
The modeling process has been broken into three steps: preprocessing, modeling, and post-processing.
![alt-tag](https://github.com/UVAdMIST/Norfolk_Groundwater_Model/blob/master/Norfolk_GWL_Workflow.png)

# Model Dependencies
The main model dependencies used are:
- Tensorflow
- Keras
- sklearn
- Pandas
- Numpy

# Useful Links
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - A great blog post by Andrea Karpathy explaining what RNNs are and why they work so well.
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) - One of the clearest explanations of LSTMs, from Christopher Olah's blog.
- [Deeplearning Videos](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A/playlists) - Educational and entertaining videos by Siraj Raval on many AI topics including tensorflow, deeplearning, LSTM, and RNN.
