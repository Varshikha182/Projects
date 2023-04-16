import io
import gradio as gr
import numpy as np
from transformers import AutoFeatureExtractor, YolosForObjectDetection
from PIL import Image
import torch
import matplotlib.pyplot as plt
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
def model_inference(img):
    model_name = "yolos-small-dwr"
    prob_threshold = 0.4 
    feature_extractor = AutoFeatureExtractor.from_pretrained(f"hustvl/{model_name}")
    model = YolosForObjectDetection.from_pretrained(f"hustvl/{model_name}")
    img = Image.fromarray(img)
    pixel_values = feature_extractor(img, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > prob_threshold
    target_sizes = torch.tensor(img.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    box_scaled = postprocessed_outputs[0]['boxes']

    res_img = plot_results(img, probas[keep], box_scaled[keep], model)
    return res_img

def plot_results(pil_img, prob, boxes, model):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        cl = p.argmax()
        object_class = model.config.id2label[cl.item()]
        
       
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color=c, linewidth=3))
        text = f'{object_class}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    return fig2img(plt.gcf())
    
def fig2img(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer)
    buffer.seek(0)
    img = Image.open(buffer)
    return img

description = """Here is the demo ... Upload any image to test"""

image_in = gr.components.Image()
image_out = gr.components.Image()


Iface2 = gr.Interface(
    fn=model_inference,
    inputs=[gr.Image("idd20k_lite/leftImg8bit/train/5/004906_image.jpg")],
    outputs=image_out,
    title="Traffic Vehicle Detection",
    description=description,
).launch(share = True)


"""
REQUIREMENTS :
absl-py==1.0.0
aiofiles==23.1.0
aiohttp==3.8.4
aiosignal==1.3.1
altair==4.2.2
anyio==3.6.2
asttokens==2.2.1
astunparse==1.6.3
async-timeout==4.0.2
attrs==21.4.0
backcall==0.2.0
cachetools==5.0.0
certifi==2021.10.8
charset-normalizer==2.0.12
click==8.1.3
colorama==0.4.4
cv==1.0.0
cvzone==1.5.6
cycler==0.11.0
decorator==5.1.1
entrypoints==0.4
executing==1.2.0
fastapi==0.92.0
ffmpy==0.3.0
filelock==3.9.0
flatbuffers==2.0
fonttools==4.30.0
frozenlist==1.3.3
fsspec==2023.1.0
gast==0.4.0
gitdb==4.0.10
GitPython==3.1.31
google-auth==2.6.6
google-auth-oauthlib==0.4.6
google-pasta==0.2.0
gradio==3.19.1
grpcio==1.46.0
h11==0.14.0
h5py==3.6.0
httpcore==0.16.3
httpx==0.23.3
huggingface-hub==0.12.1
idna==3.3
imageio==2.19.0
ipython==8.10.0
jedi==0.18.2
Jinja2==3.1.2
jsonschema==4.17.3
keras==2.11.0
Keras-Preprocessing==1.1.2
kiwisolver==1.3.2
libclang==14.0.1
linkify-it-py==1.0.3
Markdown==3.3.7
markdown-it-py==2.1.0
MarkupSafe==2.1.2
matplotlib==3.5.1
matplotlib-inline==0.1.6
mdit-py-plugins==0.3.3
mdurl==0.1.2
mediapipe==0.8.9.1
MouseInfo==0.1.3
multidict==6.0.4
networkx==2.8
numpy==1.24.1
oauthlib==3.2.0
opencv-contrib-python==4.5.5.64
opencv-python==4.5.5.64
opt-einsum==3.3.0
orjson==3.8.6
packaging==21.3
pandas==1.5.3
parso==0.8.3
pickleshare==0.7.5
Pillow==9.0.1
prompt-toolkit==3.0.37
protobuf==3.19.4
psutil==5.9.4
pure-eval==0.2.2
pyasn1==0.4.8
pyasn1-modules==0.2.8
PyAutoGUI==0.9.53
pycryptodome==3.17
pydantic==1.10.5
pydub==0.25.1
PyGetWindow==0.0.9
Pygments==2.14.0
PyMsgBox==1.0.9
pyparsing==3.0.7
pyperclip==1.8.2
PyRect==0.1.4
pyrsistent==0.19.3
PyScreeze==0.1.28
python-dateutil==2.8.2
python-multipart==0.0.5
pytweening==1.0.4
pytz==2022.7.1
PyWavelets==1.3.0
pywin32==303
PyYAML==6.0
regex==2022.10.31
requests==2.27.1
requests-oauthlib==1.3.1
rfc3986==1.5.0
rsa==4.8
scikit-image==0.19.2
scipy==1.8.0
seaborn==0.12.2
six==1.16.0
smmap==5.0.0
sniffio==1.3.0
stack-data==0.6.2
starlette==0.25.0
tensorboard==2.11.2
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow-estimator==2.11.0
tensorflow-io-gcs-filesystem==0.25.0
termcolor==1.1.0
tf-estimator-nightly==2.8.0.dev2021122109
thop==0.1.1.post2209072238
tifffile==2022.5.4
tokenizers==0.13.2
toolz==0.12.0
torch==1.13.1
torchvision==0.14.1
tqdm==4.64.0
traitlets==5.9.0
transformers==4.26.1
typing_extensions==4.2.0
uc-micro-py==1.0.1
urllib3==1.26.9
uvicorn==0.20.0
wcwidth==0.2.6
websockets==10.4
Werkzeug==2.1.2
wrapt==1.14.1
yarl==1.8.2
"""
