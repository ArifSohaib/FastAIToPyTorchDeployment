import unittest
from prepare_and_process_image import transform_image
from app import get_prediction
import requests

class FunctionTests(unittest.TestCase):
    def test_transforms(self,image = 'images/FarmaidBot2.jpg'):
        with open(image,'rb') as f:
            image_bytes = f.read()
            tensor = transform_image(image_bytes)
            print(tensor)
    def test_prediction(self, image= 'images/FarmaidBot2.jpg'):
        with open(image, 'rb') as f:
            image_bytes = f.read()
            print(get_prediction(image_bytes))
    def test_api(self):
        resp = requests.post("http://127.0.0.1:5000/predict",files={"file": open('images/FarmaidBot2.jpg','rb')})
        print(resp.json())
if __name__ == '__main__':
    unittest.main()