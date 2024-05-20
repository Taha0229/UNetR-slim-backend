from rest_framework.response import Response
from rest_framework.decorators import api_view
from .utils import encodeImageIntoBase64, decodeImage
from .predict import PredictionPipeline



@api_view(["POST"])
def run_prediction(request, *args, **kwargs):
    method = request.method
    if method == "POST":
        input_data = request.data
        
        image_name = input_data.get("imgname")
        base64_image_string = input_data.get("image")
        
        if image_name is not None and base64_image_string is not None:
            decodeImage(base64_image_string, image_name)
            predict = PredictionPipeline()
            predict.predict(filename=image_name)
            output_encoded = encodeImageIntoBase64(image_name)
            return Response({"output": output_encoded})
        else:
            return Response({"output": "Missing data in POST Request"}, status=400)