from download_weight import download

model_list = ['densenet121.ra_in1k']

for model_name in model_list:
    download(model_name)
    