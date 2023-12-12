from download_weight import download

model_list = ['seresnext101_32x4d', 'seresnext26d_32x4d', 'seresnext50_32x4d', 'convnext_large.fb_in22k_ft_in1k']

for model_name in model_list:
    download(model_name)
    