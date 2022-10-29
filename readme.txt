#Note we have assumbed the following directory structure for path to data
classification/data
regression/data
<22m0742>
├── nn_1.py
├── nn_2.py
├── nn_3.py
├── part_2.csv
├── part_3.csv
├── features.csv
├── plots_1b
│   ├── dev_32.png
│   ├── dev_64.png
│   ├── train_32.png
│   └── train_64.png
├── classification `This submission directory is optional`
│   ├── params.csv
│   └── nn_classification.py
└── readme.txt

nn_1.py Consist of code for Part 1 with no enhancements. Just simple adam optimizer with 3 layers each with 16 neurons.
        To run: python nn_1.py
        Will generate nothing

nn_2.py Consists of all enhancements like dropout, PCA, feature scaling, variable number of neurons for each layer, 
        and upsampling to create data for years which have less number of examples.
        In the above code we have implemented PCA in which we are using components instead of features. 
        So we have created PCA_betas.csv which will be generated once the code is run from which most important 
        components are weightage from each features can be determined.
        To run: python nn_2.py
        Will generate PCA_betas.csv and 22m0742.csv which contains the predictions for test data.

nn_3.py Same code as nn_2.py since feature scaling and selection was allowing in part 2. 
        This tasks asks to fill features.csv file but since PCA is used we have stored 
        the most important components in PCA_betas.csv file.
        In features.csv file we have stored names of all features since each component is a combination of all features.
        To run: python nn_3.py
        Will generate PCA_betas.csv and 22m0742.csv which contains the predictions for test data.

nn_classification.py Uses same code file as nn_2.py but with different hyperparameters. 
        This file also generated PCA_betas.csv same as nn_2.py
        To run: cd classification/ && python nn_classification.py
        Will generate PCA_betas.csv and 22m0742.csv which contains the predictions for test data.







