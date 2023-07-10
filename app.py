import base64
import numpy as np
from potassium import Potassium, Request, Response
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to("cuda")
    model = SamAutomaticMaskGenerator(sam)
    context = {"model": model}
    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    image_string = request.json.get("image")
    if image_string == None:
        return {"message": "No image provided"}

    model = context.get("model")

    img_original = base64.b64decode(image_string)
    img_as_np = np.frombuffer(img_original, dtype=np.uint8)
    image = cv2.imdecode(img_as_np, flags=1)
    masks = model.generate(image)
    for mask in masks:
        mask["segmentation"] = mask["segmentation"].tolist()

    return Response(json={"masks": masks}, status=200)


if __name__ == "__main__":
    app.serve()
