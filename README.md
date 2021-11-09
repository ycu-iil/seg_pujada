# Segmentation method to estimate the coverage of corals, seagrass, and sea urchins

This is an implenetation of U-net for classifiaction of underwater environments [1].

[1] K. Terayama, K. Mizuno*, S. Tabeta, S. Sakamoto, Y. Sugimoto, K. Sugimoto, H. Fukami, M. Sakagami, L. A. Jimenez, ["Cost-effective seafloor habitat mapping using a portable speedy sea scanner and deep-learning-based segmentation: A sea trial at Pujada Bay, Philippines,"](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13744) Methods in Ecology and Evolution, 2021. [DOI: 10.1111/2041-210X.13744]


## Requirements
- python 3.6 
- NumPy 1.13  
- tensorflow 1.4  
- Keras 2.0.8
- Pillow 7.0

## How to estimate
1. Clone this repository and move into it.

2. Load the trained model (https://doi.org/10.6084/m9.figshare.16655332). This model is trained by using the dataset of 4 lines. (see details in the paper)

3. Put images for prediction in the /test_png directory. Please prepare images with a size of 512x512 pixels and the png format. As examples, several images in the line 2 are prepared in the directory. 

4. Estimation. Please set the path of the downloaded model to the --load_model option. The estimated results are output to the test_prediction directory. 

```bash
python main_4lines.py --whole_prediction --nb_class 4 --load_model model.hdf5
```

