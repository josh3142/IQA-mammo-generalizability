# Estimating Contrast-Detail-Curves with neural networks 
Accompanying code to [About the Generalizability of Deep Learning based Image
Quality Assessment in Mammography](https://hydra.cc/docs/intro/) to generate Contrast-Detail-Curves (CDCs) to assess the image quality of mammography units.

This code contains a toy example only to generate some CDCs.

If you need further information please consult one of the authors of the corresponding paper. 

## Requirements

### Packages
Create a virtual environment with `python=3.9` and install the all the packages with `pip` from `requirement_Mammo.txt`. The relevant environment is provided:
```
conda env create -f environment/create_env_Mammography.yml
conda activate Mammo
pip install -r environment/requirement_Mammo.txt
```

## Download the dataset
A toy dataset with 17 synthetically devices can be downloaded [here](). For each device 50 images have been generated and the images were resized to 250 x 250 pixels using the resize method lanczos.

Copy the downloaded datasets into the main folder to run the script.

## Running the script
To manage different runs the package [hydra](https://hydra.cc/docs/intro/) is used. Default setting generate CDCs for different architectures resizing the images to 250 x 250 pixels using lanczos. 
```python run.py```
The default settings can be overwritten easily. To train a model on resizing method with cubic resizing and ResNet18 architecture on device cuda:0 with seed 1 can be done by
```
python run.py device=cuda:0 data=cubic arch=resnet seed=1
```

Alternatively training, plotting the loss curves and generating the CDCs can be done by running
```
bash running_script/run_toy_example.sh
``` 

## Disclaimer
This software was developed at Physikalisch-Technische Bundesanstalt
(PTB). The software is made available "as is" free of cost. PTB assumes
no responsibility whatsoever for its use by other parties, and makes no
guarantees, expressed or implied, about its quality, reliability, safety,
suitability or any other characteristic. In no event will PTB be liable
for any direct, indirect or consequential damage arising in connection

## License
MIT License

Copyright (c) 2023 Datenanalyse und Messunsicherheit, working group 8.42, PTB Berlin

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Reference
