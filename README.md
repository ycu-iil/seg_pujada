# Segmentation method to estimate the coverage of corals, seagrass, and sea urchins

This is an implenetation of U-net for classifiaction of underwater environments.

## Requirements
- python 3.6 
- NumPy 1.13  
- tensorflow 1.4  
- Keras 2.0.8
- Pillow 7.0

## How to estimate
1. Clone this repository and move into it.

2. Load the trained model. We uploaded it at XXX. This model is trained by using the dataset of 4 lines. (see details in the paper)

3. Put images for prediction in the /test_png directory. As examples, several images in the line 2 are prepared in the directory. 

4. Estimation. Please set the path of the downloaded model to the --load_model option. 

```bash
python main_4lines.py --whole_prediction --nb_class 4 --load_model model.hdf5
```

