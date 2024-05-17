from rest_framework.response import Response
from rest_framework.decorators import api_view
from .utils import encodeImageIntoBase64, decodeImage
from .predict import PredictionPipeline



@api_view(["GET", "POST"])
def run_prediction(request, *args, **kwargs):
    method = request.method
    if method == "GET":

        return Response({"req": "get req"})

    if method == "POST":
        input_data = request.data
        
        file_name = input_data["imgname"]    
        decodeImage(input_data["image"], file_name)
        predict = PredictionPipeline()
        predict.predict(filename=file_name)
        output_encoded = encodeImageIntoBase64(file_name)
        return Response({"output": output_encoded})