#!/usr/bin/env python
# coding: utf-8

# # 0. Setup Paths

# In[1]:


from IPython import get_ipython

# In[2]:
def numberplatedetection():
    import os

    CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
    PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'label_map.pbtxt'


    # In[3]:


    paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('ANPR','Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('ANPR','Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
}

    files = {
        'PIPELINE_CONFIG':os.path.join('ANPR','Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join('ANPR',paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
        'LABELMAP': os.path.join('ANPR',paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }


    # In[26]:


    # for path in paths.values():
    #     if not os.path.exists(path):
    #         if os.name == 'posix':
    #             !mkdir -p {path}
    #         if os.name == 'nt':
    #             !mkdir {path}


    # # 1. Download TF Models Pretrained Models from Tensorflow Model Zoo and Install TFOD

    # In[ ]:


    # https://www.tensorflow.org/install/source_windows


    # In[5]:


    if os.name=='nt':
    #     !pip install wget
        import wget


    # In[28]:


    # if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    #     !git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}


    # In[29]:


    # # Install Tensorflow Object Detection 
    # if os.name=='posix':  
    #     !apt-get install protobuf-compiler
    #     !cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install . 
        
    # if os.name=='nt':
    #     url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
    #     wget.download(url)
    #     !move protoc-3.15.6-win64.zip {paths['PROTOC_PATH']}
    #     !cd {paths['PROTOC_PATH']} && tar -xf protoc-3.15.6-win64.zip
    #     os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))   
    #     !cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install
    #     !cd Tensorflow/models/research/slim && pip install -e . 


    # In[ ]:


    # !pip list


    # In[9]:


    # VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
    # # Verify Installation
    # !python {VERIFICATION_SCRIPT}


    # In[ ]:


    # !pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 --upgrade


    # In[ ]:


    # !pip uninstall protobuf matplotlib -y
    # !pip install protobuf matplotlib==3.2


    # In[ ]:


    # !pip install Pillow


    # In[ ]:


    # !pip install pyyaml


    # In[ ]:


    # !pip list


    # In[8]:


    import object_detection


    # In[32]:


    # if os.name =='posix':
    #     !wget {PRETRAINED_MODEL_URL}
    #     !mv {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}
    #     !cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}
    # if os.name == 'nt':
    #     wget.download(PRETRAINED_MODEL_URL)
    #     !move {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}
    #     !cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}


    # # 2. Create Label Map

    # In[ ]:


    # labels = [{'name':'licence', 'id':1}]

    # with open(files['LABELMAP'], 'w') as f:
    #     for label in labels:
    #         f.write('item { \n')
    #         f.write('\tname:\'{}\'\n'.format(label['name']))
    #         f.write('\tid:{}\n'.format(label['id']))
    #         f.write('}\n')


    # # 3. Create TF records

    # In[ ]:


    # # OPTIONAL IF RUNNING ON COLAB
    # ARCHIVE_FILES = os.path.join(paths['IMAGE_PATH'], 'archive.tar.gz')
    # if os.path.exists(ARCHIVE_FILES):
    #   !tar -zxvf {ARCHIVE_FILES}


    # In[ ]:


    # if not os.path.exists(files['TF_RECORD_SCRIPT']):
    #     !git clone https://github.com/nicknochnack/GenerateTFRecord {paths['SCRIPTS_PATH']}


    # In[ ]:


    # !pip install pytz


    # In[ ]:


    # !python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'train')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')} 
    # !python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'test')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')} 


    # # 4. Copy Model Config to Training Folder

    # In[ ]:


    # if os.name =='posix':
    #     !cp {os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'])}
    # if os.name == 'nt':
    #     !copy {os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'])}


    # # 5. Update Config For Transfer Learning

    # In[6]:


    # import tensorflow as tf
    # from object_detection.utils import config_util
    # from object_detection.protos import pipeline_pb2
    # from google.protobuf import text_format


    # In[ ]:


    # config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])


    # In[ ]:


    # config


    # In[ ]:


    # pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    # with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
    #     proto_str = f.read()                                                                                                                                                                                                                                          
    #     text_format.Merge(proto_str, pipeline_config)  


    # In[ ]:


    # pipeline_config.model.ssd.num_classes = len(labels)
    # pipeline_config.train_config.batch_size = 4
    # pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    # pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    # pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
    # pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
    # pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
    # pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]


    # In[ ]:


    # config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
    # with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
    #     f.write(config_text)   


    # # 6. Train the model

    # In[ ]:


    # TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')


    # In[ ]:


    # command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=10000".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])


    # In[ ]:


    # print(command)


    # In[ ]:


    # !{command}


    # # 7. Evaluate the Model

    # In[ ]:


    # command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])


    # In[ ]:


    # print(command)


    # In[ ]:


    # !{command}


    # # 8. Load Train Model From Checkpoint

    # In[10]:


    import os
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.builders import model_builder
    from object_detection.utils import config_util


    # In[11]:


    # Prevent GPU complete consumption
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try: 
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
        except RunTimeError as e:
            print(e)


    # In[12]:


    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections


    # # 9. Detect from an Image

    # In[13]:


    import cv2 
    import numpy as np
    from matplotlib import pyplot as plt
    # get_ipython().run_line_magic('matplotlib', 'inline')


    # In[14]:


    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])


    # In[15]:


    # IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test','Cars428.png')


    # In[16]:


    # img = cv2.imread(IMAGE_PATH)
    # image_np = np.array(img)

    # input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    # detections = detect_fn(input_tensor)

    # num_detections = int(detections.pop('num_detections'))
    # detections = {key: value[0, :num_detections].numpy()
    #               for key, value in detections.items()}
    # detections['num_detections'] = num_detections

    # # detection_classes should be ints.
    # detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # label_id_offset = 1
    # image_np_with_detections = image_np.copy()

    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #             image_np_with_detections,
    #             detections['detection_boxes'],
    #             detections['detection_classes']+label_id_offset,
    #             detections['detection_scores'],
    #             category_index,
    #             use_normalized_coordinates=True,
    #             max_boxes_to_draw=5,
    #             min_score_thresh=.8,
    #             agnostic_mode=False)

    # plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    # plt.show()


    # In[108]:


    # detections.keys()


    # # Apply OCR to Detection

    # In[ ]:


    # !pip install easyocr


    # In[ ]:


    # !pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html


    # In[17]:


    import easyocr


    # In[18]:


    detection_threshold = 0.7


    # In[19]:


    # image = image_np_with_detections
    # scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    # boxes = detections['detection_boxes'][:len(scores)]
    # classes = detections['detection_classes'][:len(scores)]


    # In[20]:


    # width = image.shape[1]
    # height = image.shape[0]


    # In[21]:


    # Apply ROI filtering and OCR
    # for idx, box in enumerate(boxes):
    #     print(box)
    #     roi = box*[height, width, height, width]
    #     print(roi)
    #     region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
    #     plt.imshow(region)
    #     reader = easyocr.Reader(['en'])
    #     ocr_result = reader.readtext(region)
    #     print(ocr_result)
    #     plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))


    # In[22]:


    # for result in ocr_result:
    #     print(np.sum(np.subtract(result[0][2],result[0][1])))
    #     print(result[1])


    # # OCR Filtering

    # In[23]:


    region_threshold = 0.05


    # In[24]:


    def filter_text(region, ocr_result, region_threshold):
        rectangle_size = region.shape[0]*region.shape[1]
        
        plate = [] 
        for result in ocr_result:
            length = np.sum(np.subtract(result[0][1], result[0][0]))
            height = np.sum(np.subtract(result[0][2], result[0][1]))
            
            if length*height / rectangle_size > region_threshold:
                plate.append(result[1])
        return plate


    # In[25]:


    # filter_text(region, ocr_result, region_threshold)


    # # Bring it Together

    # In[93]:


    # region_threshold = 0.6


    # In[26]:


    def ocr_it(image, detections, detection_threshold, region_threshold):
        
        # Scores, boxes and classes above threhold
        scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
        boxes = detections['detection_boxes'][:len(scores)]
        classes = detections['detection_classes'][:len(scores)]
        
        # Full image dimensions
        width = image.shape[1]
        height = image.shape[0]
        
        # Apply ROI filtering and OCR
        for idx, box in enumerate(boxes):
            roi = box*[height, width, height, width]
            region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
            reader = easyocr.Reader(['en'])
            ocr_result = reader.readtext(region)
            
            text = filter_text(region, ocr_result, region_threshold)
            
            plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            plt.show()
            print(text)
            return text, region


    # In[27]:


    # text, region = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)


    # # Save Results

    # In[1]:


    import csv
    import uuid
    

    # In[2]:


    '{}.jpg'.format(uuid.uuid1())


    # In[3]:


    def save_results(text, region, csv_filename, folder_path):
        img_name = '{}.jpg'.format(uuid.uuid1())
        
        cv2.imwrite(os.path.join(folder_path, img_name), region)
        
        with open(csv_filename, mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([img_name, text])
        return img_name


    # In[4]:


    # region


    # In[5]:


    # save_results(text, region, 'detection_results.csv', 'Detection_Images')


    # # 10. Real Time Detections from your Webcam

    # In[ ]:


    # !pip uninstall opencv-python-headless -y


    # In[28]:


    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened(): 
        ret, frame = cap.read()
        image_np = np.array(frame)
        
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.8,
                    agnostic_mode=False)
        
        try: 
            text, region = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)
            print(type(text[0]))
            print(text[0])
            imagename=save_results(text, region, 'datection_results.csv', '../static/media')
            cap.release()
            cv2.destroyAllWindows()
            try:
                return text[0]+text[1],imagename
            except:
                return text[0],imagename
        except:
            pass

        cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


    # # 10. Freezing the Graph

    # In[ ]:


    # FREEZE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'exporter_main_v2.py ')


    # In[ ]:


    # command = "python {} --input_type=image_tensor --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(FREEZE_SCRIPT ,files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'], paths['OUTPUT_PATH'])


    # In[ ]:


    # print(command)


    # In[ ]:


    # get_ipython().system('{command}')


    # # 11. Conversion to TFJS

    # In[ ]:


    # !pip install tensorflowjs


    # In[ ]:


    # command = "tensorflowjs_converter --input_format=tf_saved_model --output_node_names='detection_boxes,detection_classes,detection_features,detection_multiclass_scores,detection_scores,num_detections,raw_detection_boxes,raw_detection_scores' --output_format=tfjs_graph_model --signature_name=serving_default {} {}".format(os.path.join(paths['OUTPUT_PATH'], 'saved_model'), paths['TFJS_PATH'])


    # In[ ]:


    # print(command)


    # In[ ]:


    # !{command}


    # In[ ]:


    # Test Code: https://github.com/nicknochnack/RealTimeSignLanguageDetectionwithTFJS


    # # 12. Conversion to TFLite

    # In[ ]:


    # TFLITE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'export_tflite_graph_tf2.py ')


    # In[ ]:


    # command = "python {} --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(TFLITE_SCRIPT ,files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'], paths['TFLITE_PATH'])


    # In[ ]:


    # print(command)


    # In[ ]:


    # !{command}


    # In[ ]:


    # FROZEN_TFLITE_PATH = os.path.join(paths['TFLITE_PATH'], 'saved_model')
    # TFLITE_MODEL = os.path.join(paths['TFLITE_PATH'], 'saved_model', 'detect.tflite')


    # In[ ]:


    # command = "tflite_convert \
    # --saved_model_dir={} \
    # --output_file={} \
    # --input_shapes=1,300,300,3 \
    # --input_arrays=normalized_input_image_tensor \
    # --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
    # --inference_type=FLOAT \
    # --allow_custom_ops".format(FROZEN_TFLITE_PATH, TFLITE_MODEL, )


    # In[ ]:


    # print(command)


    # In[ ]:


    # !{command}


    # # 13. Zip and Export Models 

    # In[ ]:


    # !tar -czf models.tar.gz {paths['CHECKPOINT_PATH']}


    # In[ ]:


    # from google.colab import drive
    # drive.mount('/content/drive')

