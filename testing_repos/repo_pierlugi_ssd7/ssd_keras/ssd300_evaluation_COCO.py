#!/usr/bin/env python
# coding: utf-8

# # SSD300 MS COCO Evaluation Tutorial
# 
# This is a brief tutorial that goes over how to evaluate a trained SSD300 on one of the MS COCO datasets using the official MS COCO Python tools available here:
# 
# https://github.com/cocodataset/cocoapi
# 
# Follow the instructions in the GitHub repository above to install the `pycocotools`. Note that you will need to set the path to your local copy of the PythonAPI directory in the subsequent code cell.
# 
# Of course the evaulation procedure described here is identical for SSD512, you just need to build a different model.

# In[ ]:


from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt
import sys

# TODO: Specify the directory that contains the `pycocotools` here.
pycocotools_dir = '../cocoapi/PythonAPI/'
if pycocotools_dir not in sys.path:
    sys.path.insert(0, pycocotools_dir)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.coco_utils import get_coco_category_maps, predict_all_to_json

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Set the input image size for the model.
img_height = 300
img_width = 300


# ## 1. Load a trained SSD
# 
# Either load a trained model or build a model and load trained weights into it. Since the HDF5 files I'm providing contain only the weights for the various SSD versions, not the complete models, you'll have to go with the latter option when using this implementation for the first time. You can then of course save the model and next time load the full model directly, without having to build it.
# 
# You can find the download links to all the trained model weights in the README.

# ### 1.1. Build the model and load trained weights into it

# In[4]:


# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=80,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05], # The scales for Pascal VOC are [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.01,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
weights_path = 'path/to/trained/weights/VGG_coco_SSD_300x300_iter_400000.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


# Or

# ### 1.2. Load a trained model

# In[ ]:


# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = 'path/to/trained/model.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})


# ## 2. Create a data generator for the evaluation dataset
# 
# Instantiate a `DataGenerator` that will serve the evaluation dataset during the prediction phase.

# In[5]:


dataset = DataGenerator()

# TODO: Set the paths to the dataset here.
MS_COCO_dataset_images_dir = '../../datasets/MicrosoftCOCO/val2017/'
MS_COCO_dataset_annotations_filename = '../../datasets/MicrosoftCOCO/annotations/instances_val2017.json'

dataset.parse_json(images_dirs=[MS_COCO_dataset_images_dir],
                   annotations_filenames=[MS_COCO_dataset_annotations_filename],
                   ground_truth_available=False, # It doesn't matter whether you set this `True` or `False` because the ground truth won't be used anyway, but the parsing goes faster if you don't load the ground truth.
                   include_classes='all',
                   ret=False)

# We need the `classes_to_cats` dictionary. Read the documentation of this function to understand why.
cats_to_classes, classes_to_cats, cats_to_names, classes_to_names = get_coco_category_maps(MS_COCO_dataset_annotations_filename)


# ## 3. Run the predictions over the evaluation dataset
# 
# Now that we have instantiated a model and a data generator to serve the dataset, we can make predictions on the entire dataset and save those predictions in a JSON file in the format in which COCOeval needs them for the evaluation.
# 
# Read the documenation to learn what the arguments mean, but the arguments as preset below are the parameters used in the evaluation of the original Caffe models.

# In[6]:


# TODO: Set the desired output file name and the batch size.
results_file = 'detections_val2017_ssd300_results.json'
batch_size = 20 # Ideally, choose a batch size that divides the number of images in the dataset.


# In[7]:


predict_all_to_json(out_file=results_file,
                    model=model,
                    img_height=img_height,
                    img_width=img_width,
                    classes_to_cats=classes_to_cats,
                    data_generator=dataset,
                    batch_size=batch_size,
                    data_generator_mode='resize',
                    model_mode='inference',
                    confidence_thresh=0.01,
                    iou_threshold=0.45,
                    top_k=200,
                    normalize_coords=True)


# ## 4. Run the evaluation
# 
# Now we'll load the JSON file containing all the predictions that we produced in the last step and feed it to `COCOeval`. Note that the evaluation may take a while.

# In[8]:


coco_gt   = COCO(MS_COCO_dataset_annotations_filename)
coco_dt   = coco_gt.loadRes(results_file)
image_ids = sorted(coco_gt.getImgIds())


# In[9]:


cocoEval = COCOeval(cocoGt=coco_gt,
                    cocoDt=coco_dt,
                    iouType='bbox')
cocoEval.params.imgIds  = image_ids
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()


# In[ ]:




