<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/sommaa/CoverType-prediction">
    <img src="https://github.com/sommaa/CoverType-prediction/assets/120776791/e27f51b9-4782-402e-ab52-4d2319bdff26" alt="Logo" width="140" height="140">
  </a>
      <br />
    <h1 align="center">CoverType prediction</h3>
    <h4 align="center">Predicting forest cover type from cartographic variables only.</h4>
  
</div>

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

</div>

Authors: [Andrea Somma](https://github.com/sommaa), [Lorenzo Paggetta](https://github.com/lpaggetta), [Pietro Marelli](https://github.com/Pietro-Marelli) 

The present work showcases different methods to develop a classifier for the Cover Type dataset,
in order to achieve an accurate and balanced model for the cover forest type from the cartographic
variables in the dataset.
The Cover Type dataset contains trees observation from four wilderness areas of the Roosevelt
National forest in Colorado. The data is made of cartographic variables only, with no remotely
sensed data. It is a rather large dataset, made of 7 forest cover types, more than half a
million instances and 54 features, which include data such as elevation, aspect, slope,
distance to hydrology, soil type and many others.

<div align="center">
  
  ## :bow_and_arrow: Targets
  
  |      | Forest Cover Type    |
  |------|----------------------|
  | 1    | Spruce/Fir           |
  | 2    | Lodgepole Pine       |
  | 3    | Ponderosa Pine       |
  | 4    | Cottonwood/Willow    |
  | 5    | Aspen                |
  | 6    | Douglas-fir          |
  | 7    | Krummholz            |

  ## :books: Features
  
  | Label Code | Label Type                        | Data Type |
  |------------|-----------------------------------|-----------|
  | 1          | Elevation                         | Integer   |
  | 2          | Aspect                            | Integer   |
  | 3          | Slope                             | Integer   |
  | 4          | Horizontal Distance To Hydrology  | Integer   |
  | 5          | Vertical Distance To Hydrology    | Integer   |
  | 6          | Horizontal Distance To Roadways   | Integer   |
  | 7          | Hillshade 9am                     | Integer   |
  | 8          | Hillshade Noon                    | Integer   |
  | 9          | Hillshade 3pm                     | Integer   |
  | 10         | Horizontal Distance To Fire Points| Integer   |
  | 11-14      | Wilderness Area                   | Binary    |
  | 15-54      | Soil Type                         | Binary    |

  ## Data classes

  ![Intro_bar_datapoints_covtype](https://github.com/sommaa/CoverType-prediction/assets/120776791/9dd9b472-3a53-47df-9ad9-71558f7a2eb7)
  
  ## :racehorse: Performance ML
  
  | Model                  | Accuracy [%] | Parameters | Size [MB] | Training Time   |
  |------------------------|--------------|------------|-----------|-----------------|
  | Bagging-based - Rescaled    | 97           | 3.9M       | 24        | 5 min           |
  | DecisionTree-based - Rescaled   | 92           | 6k         | 3         | 2 min           |
  | DecisionTree-based opt - Rescaled | 90           | 3k         | 0.72      | ~20 seconds     |

  ## :racehorse: Performance NN

  | Model                | Accuracy [%] | Parameters | Size [kB] | Training Time   |
  |----------------------|--------------|------------|-----------|-----------------|
  | NN - Rescaled               | 93.3         | 233.9k     | 2850      | 9 min           |
  | NN opt - non-quantized - Rescaled     | 90.3         | 10.6k      | 172       | 4 min           |
  | NN opt - quantized - Rescaled      | 90           | 10.6k      | 19.5      | 4 min           |

  ## :classical_building: NN Acrhitecture

  ### Convolutional Neural Network (NN)
  
  ![visNN-1-1_page-0001](https://github.com/sommaa/CoverType-prediction/assets/120776791/7a787e01-6c56-4eda-885f-bbf1466299eb)

  ### Convolutional Neural Network Optimized (NN opt)
  
  ![visNNSmall-1_page-0001](https://github.com/sommaa/CoverType-prediction/assets/120776791/15792165-f402-4d94-83ef-52fe35e5857d)

</div>
