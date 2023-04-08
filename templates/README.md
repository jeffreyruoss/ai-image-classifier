# Image Classification Web App

This web app uses the VGG16 pre-trained model to classify images.

## Requirements

- Python 3.6 or higher
- Flask
- TensorFlow


## How it works

The app uses the Flask web framework to create a simple web interface for uploading images. The uploaded images are processed using the VGG16 pre-trained model from TensorFlow. The pre-trained model is used to predict the class of the image, and the predicted class name is then returned and displayed in the web interface.

### Main files

- `app.py`: The main Flask application file that contains the server-side code for handling image uploads and classification.
- `templates/index.html`: The HTML template for the web interface, which includes the form for uploading images and the JavaScript code for making AJAX requests to the server.
