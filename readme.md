# PyTorch Web API deployment

This repo shows you how to deploy a model trained in FastAI to a web API using only PyTorch.
First follow the given notebook to remove the FastAI dependency.

The repo comes with a model trained to recognise 5 types of traffic signs as an example and this model was trained using FastAI.

To run the website locally use a command line tool to navigate to the folder where you downloaded or cloned this repo, then type:
`python app.py`

This will start up a Flask server that you can use via a UI or via an API.

To verify everything is working run:
`python tests.py`

To change the model to yours, change the `static\pytorch_traffic_sign` file with your own pytorch model.
If you are using a model originally trained with FastAI, make sure to verify that you have replaced the head of the model properly using the instructions in the Google Colab notebook.
<br\>
After changing the model, also create a json file with the names of all categories your trained on. For reference check `static\traffic_class.json`

## requirements

The app and API requires PyTorch, TorchVision and Flask only, it has been tested with the following version  

- Flask==1.0.3
- PyTorch==1.3.0
- Torchvision==0.4.0


The code here is based on the PyTorch documentation<a href="https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html"> here</a> as well as a linked repository <a href="https://github.com/avinassh/pytorch-flask-api-heroku">here</a>.