from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


config_file = 'configs/rugd/rugdconfig.py'
checkpoint_file = 'work_dirs/BC_RUGD_2000epoch_6batch/epoch_1700.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
# test a single image
# img = 'RUGD/RUGD_raw/right0157.png'
img='road.jpg'
result = inference_segmentor(model, img)
# show the results
show_result_pyplot(model, img, result, get_palette('RUGDDataset'))