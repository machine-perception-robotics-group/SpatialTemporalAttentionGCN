## Requirement
- pytorch  1.0.1


# Data Preparing

## download
- [NTU-RGB+D60](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp)    
Only the 3D skeleton (body joints) is required.   
If you want to use the dataset, you need to apply for it, but only the skeleton data is available on [Github](https://github.com/shahroudy/NTURGB-D).   

- [NTU-RGB+D120](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp)   
Only the 3D skeleton (body joints) is required.  
Only skeleton data is available from [Github](https://github.com/shahroudy/NTURGB-D)
    - The NTU-RGB+D120 can be combined with the NTU-RGB+D60 to form a complete dataset. Copy the NTU-RGB+D60 data into the NTU-RGB+D120 directory.  
    ```
    find ntu-rgb+d-skeletons60/ -name "*.skeleton" -print0 | xargs -0 -I {} mv {} ntu-rgb+d-skeletons120/
    ```

## train data, test data
Each program has a variable `origin_path`(Path of downloaded data), so change it appropriately. 
- NTU-RGB+D60  
```python Tools/Gen_dataset/ntu60.py```  

- NTU-RGB+D120  
```python Tools/Gen_dataset/ntu120.py```  

## Generate multi-modal data
- NTU-RGB+D60  
```python Tools/Gen_dataset/multi_modal.py --dataset ntu60```  

- NTU-RGB+D120  
```python Tools/Gen_dataset/multi_modal.py --dataset ntu120```



# Train
- Training and model settings are described in the config file.  
  The config files for each dataset are located in `Tools/Config/`.  
  You can change the configuration by editing the config file.

- Load the config file and train the model.
    - Cross subject : Train the model with NTU-RGB+D60 coordinate (joint coordinate, bone) data. 
    ```
    python train.py --config Tools/Config/NTU-RGB+D60/xsub/coordinate.yaml
    ```
    
    - Cross subject : Train the model with NTU-RGB+D60 velocity (joint velocity, bone velocity) data.
    ```
    python train.py --config Tools/Config/NTU-RGB+D60/xsub/velocity.yaml
    ```
    
    - Cross subjcet : Train the model with NTU-RGB+D60 acceleration (joint acceleration, bone acceleration) data.
    ```
    python train.py --config Tools/Config/NTU-RGB+D60/xsub/acceleration.yaml
    ```
    
- If you want to train with another dataset, load a different config file.   
    - Example:  
        - Cross view : NTU-RGB+D60 coordinate (joint coordinate, bone) data.  
          `--config Tools/Config/NTU-RGB+D60/xview/coordinate.yaml`
        - Cross setup : NTU-RGB+D120 velocity (joint velocity, bone velocity) data.  
          `--config Tools/Config/NTU-RGB+D120/xsetup/velocity.yaml`
       
          
# Test
- Test the trained model. Load the same config file as the training one.  

    - Cross subject : Test the model with NTU-RGB+D60 coordinate (joint coordinate, bone) data.  

    ```
    python test.py --config Tools/Config/NTU-RGB+D60/xsub/coordinate.yaml
    ```
    
    - Cross subject : Test the model with NTU-RGB+D60 velocity (joint velocity, bone velocity) data.  
    ```
    python test.py --config Tools/Config/NTU-RGB+D60/xsub/velocity.yaml
    ```
    
    - Cross subject : Test the model with NTU-RGB+D60 acceleration (joint acceleration, bone acceleration) data.  
    ```
    python test.py --config Tools/Config/NTU-RGB+D60/xsub/acceleration.yaml
    ```

- Add ``--visualize`` to plot an attention graph.   
  Plotting Attention graphs can take a long time. If you only want to evaluate the results, it is recommended not to use the ``--visualize``.     
    - Example:  
        ```
        python test.py --config Tools/Config/NTU-RGB+D60/xsub/coordinate.yaml --visualize
        ```
  
  
        
# Mechanics-stream structure
- Evaluation generates a `test_score.pkl` in each log directory.    
  Load the `test_score.pkl` and make the final inference value be the sum of the class probabilities.  
  ``` 
  python ensemble.py --dataset ntu60_xsub --score1 coordinate --score2 velocity --score3 acceleration
  ```
    - `--dataset`:Select a dataset to evaluate. Select from the following.  
    ntu60_xsub, ntu60_xview, ntu120_xsub, ntu120_xsetup
    - `--score1(2,3)`:Give the name of the directory where the `test_score.pkl` is stored.   
    If you want to evaluate the stream structure on two networks, you give two directory names. 
  