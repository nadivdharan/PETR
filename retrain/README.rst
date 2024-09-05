===============
PETR Retraining
===============

Prerequisites
-------------


* docker (\ `installation instructions <https://docs.docker.com/engine/install/ubuntu/>`_\ )
* nvidia-docker2 (\ `installation instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_\ )

**NOTE:**\  In case you are using the Hailo Software Suite docker, make sure to run all of the following instructions outside of that docker.


Environment Preparations
------------------------

#. Build the docker image:

   .. raw:: html
      :name:validation

      <pre><code stage="docker_build">
      cd <span val="dockerfile_path">hailo_model_zoo/training/petr</span>
      docker build -t petr:v2 --build-arg timezone=`cat /etc/timezone` .
      </code></pre>

   | the following optional arguments can be passed via --build-arg:

   * ``timezone`` - a string for setting up timezone. E.g. "Asia/Jerusalem"
   * ``user`` - username for a local non-root user. Defaults to 'hailo'.
   * ``group`` - default group for a local non-root user. Defaults to 'hailo'.
   * ``uid`` - user id for a local non-root user.
   * ``gid`` - group id for a local non-root user.


#. Start your docker:

   .. raw:: html
      :name:validation

      <code stage="docker_run">
      docker run <span val="replace_none">--name "your_docker_name"</span> -it --gpus all --shm-size 32gb <span val="replace_none">-u "username"</span> --ipc=host -v <span val="local_vol_path">/path/to/local/data/dir</span>:<span val="docker_vol_path">/path/to/docker/data/dir</span>  petr:v2
      </code>

   * ``docker run`` create a new docker container.
   * ``--name <your_docker_name>`` name for your container.
   * ``-it`` runs the command interactively.
   * ``--gpus all`` allows access to all GPUs.
   * ``--shm-size`` container shared memory size 
   * ``--ipc=host`` sets the IPC mode for the container.
   * ``-v /path/to/local/data/dir:/path/to/docker/data/dir`` maps ``/path/to/local/data/dir`` from the host to the container. You can use this command multiple times to mount multiple directories.
   * ``petr:v2`` the name of the docker image.
   
   .. raw:: html
      :name:validation
      <code stage="docker_run">
      docker start "your_docker_name"
      docker exec -it "your_docker_name" /bin/bash --login
      </code>

Training and exporting to ONNX
------------------------------

#. | Prepare your data: 

   | Data is expected to be in NuScenes format. For more information on obtaining datasets see `here <https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/data_preparation.md>`_
   | The expected structure is as follows:

   .. code-block::

       /workspace
       |-- PETR
       `-- |-- data
           `-- |-- nuscenes
               |   |-- maps
               |   |-- samples
               |   |-- sweeps
               |   |-- v1.0-trainval
               |   |-- mmdet3d_nuscenes_30f_infos_val.pkl
               |   |-- mmdet3d_nuscenes_30f_infos_train.pkl


   The path for the dataset can be configured in the .py config file, e.g. ``projects/configs/petrv2/petrv2_fcos3d_repvgg_b0x32_BN_q_304_decoder_3_UN_800x320.py``. It is recommended to symlink to the dataset to ``/workspace/PETR/data/``.

#. Training:

   Configure your model in a .py config file. We will use ``projects/configs/petrv2/petrv2_fcos3d_repvgg_b0x32_BN_q_304_decoder_3_UN_800x320.py`` as the config file in this guide.
   Start training with the following command:

   .. raw:: html
      :name:validation

      <pre><code stage="retrain">
      cd /workspace/PETR
      ./tools/dist_train.sh projects/configs/petrv2/petrv2_fcos3d_repvgg_b0x32_BN_q_304_decoder_3_UN_800x320.py <span val="gpu_num">4</span> --work-dir work_dirs/petrv2_exp0/
      </code></pre>

   Where 4 is the number of GPUs used for training. In this example, the trained model will be saved under ``work_dirs/petrv2_exp0/latest.pth`` directory.

#. Export to onnx

   Run the following script to export the backbone part of the model:

   .. raw:: html
      :name:validation

      <pre><code stage="export">
      cd /workspace/PETR
      python tools/export_onnx.py <cfg.py> <trained.pth> --split backbone --out petrv2_backbone.onnx
      </code></pre>

      Run the following script to export the transformer part of the model:

      <pre><code stage="export">  
      python tools/export_onnx.py <cfg.py> <trained.pth> --split transformer --out petrv2_transformer.onnx --reshape-cfg tools/onnx_reshape_cfg_repvgg_b0x32_BN2D_decoder_3_q_304_UN_800x320.json
      </code></pre>
    
   * | ``cfg.py`` - model config file path e.g., ``projects/configs/petrv2/petrv2_fcos3d_repvgg_b0x32_BN_q_304_decoder_3_UN_800x320.py``
   * | ``trained.pth`` - the trained model file path e.g., ``work_dirs/petrv2_exp0/latest.pth``
   * | ``--split`` - backbone or transformer export
   * | ``--out`` - output onnx file path
   * | ``--reshape-cfg`` - .json file with node names and config info for further reshape of the transformer export e.g., ``tools/onnx_reshape_cfg_repvgg_b0x32_BN2D_decoder_3_q_304_UN_800x320.json`` for the model we use here

   .. **NOTE:**\  Exporting the transformer also produces the ``reference_points.npy`` postprocessing configuration file.

#. Generate calibration sets

   Run the following script to generate calibration sets for the backbone (.npy) and transformer (.npz) models:

   .. raw:: html
      
      <pre><code>
      cd /workspace/PETR
      python tools/gen_calib_set.py <cfg.py> <trained.pth> --calib-set-size 64 --save-dir <save_dir> --net-name petrv2_repvggB0_transformer_pp_800x320
      </code></pre>

   * | ``--calib-set-size`` size of calibration set
   * | ``--save-dir`` path to folder to save calibration sets
   * | ``--net-name`` name of model in Hailo Model Zoo


----

Compile the Model using Hailo Model Zoo
---------------------------------------

| You can generate an HEF file for inference on Hailo-8 from your trained ONNX model.
| In order to do so you need a working model-zoo environment.
| Choose the corresponding YAMLs from our networks configuration directory, i.e. ``hailo_model_zoo/cfg/networks/petrv2_repvggB0_transformer_pp_800x320.yaml``\ and run parsing, optimization and compilation using the model zoo. 

|
* Backbone
|
.. raw:: html
   :name:validation

   <code stage="compile">
   hailomz compile --ckpt <span val="local_path_to_onnx">petrv2_backbone.onnx</span> --calib-path <span val="calib_set_path">/path/to/calibration/imgs/dir/</span> --yaml <span val="yaml_file_path">path/to/petrv2_repvggB0_backbone_pp_800x320.yaml</span> <span val="replace_none">--start-node-names name1 name2</span> <span val="replace_none">--end-node-names name1</span>
   </code>


   * | ``--ckpt`` - path to your ONNX file.
   * | ``--calib-path`` - path to a directory with your calibration images in JPEG/png format
   * | ``--yaml`` - path to your configuration YAML file.
   * | ``--start-node-names`` and ``--end-node-names`` - node names for customizing parsing behavior (optional).
   * | The model zoo will take care of adding the input normalization to be part of the model.

|
* Transformer
|

   * Parsing 
   .. raw:: html
      :name:validation

      <code stage="parse">
      hailomz parse --ckpt <span val="local_path_to_onnx">petrv2_transformer.onnx</span> --yaml <span val="yaml_file_path">path/to/petrv2_repvggB0_transformer_pp_800x320.yaml</span>
      </code>

   * | ``--ckpt`` - path to your ONNX file.
   * | ``--yaml`` - path to your configuration YAML file

   * Optimization
   .. raw:: html
      :name:validation

      <code stage="optimize">
      hailo optimize --har petrv2_repvggB0_transformer_pp_800x320.har --model-script </path/to/petrv2_repvggB0_transformer_pp_800x320.alls> --calib-set-path </path/to/transformer_calib_set>
      </code>

   * | ``--har`` - path to your parsed HAR file from the pervious step.
   * | ``--calib-set-path`` - path to transformer calibration set generated in the calibration stage above
   * | ``--model-script`` - path to model script for optimization

   * Compilation
   .. raw:: html
      :name:validation

      <code stage="compile">
      hailomz compile --har petrv2_repvggB0_transformer_pp_800x320_optimized.har --calib-path <span val="calib_set_path">/path/to/calibration/dir/</span> --yaml <span val="yaml_file_path">path/to/petrv2_repvggB0_transformer_pp_800x320.yaml</span>
      </code>

   * | ``--har`` - path to your optimized HAR file from the pervious step.
   * | ``--yaml`` - path to your configuration YAML file


.. note::
  More details about YAML files are presented `here <../../docs/YAML.rst>`_.
