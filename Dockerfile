FROM python:3.7

WORKDIR /opt
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install -r requirements.txt
# install required dep for opencv
RUN apt-get update && apt-get install libgl1 -y

COPY drusen_segmentation_weights drusen_segmentation_weights
COPY run_drusen_seg.py run_drusen_seg.py
COPY test_main_model.ipynb test_main_model.ipynb
COPY test_segmentation_Deep_lab.ipynb test_segmentation_Deep_lab.ipynb
COPY unet/ unet/

ENTRYPOINT ["python", "run_drusen_seg.py"]