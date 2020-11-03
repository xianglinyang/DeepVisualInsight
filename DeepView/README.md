# DeepView - Classification visualization

This is an implementation of the DeepView framework that was presented in the paper:  
*[Schulz, A., Hinder, F., & Hammer, B. (2020)](https://www.ijcai.org/Proceedings/2020/319). DeepView: Visualizing Classification Boundaries of Deep Neural Networks as Scatter Plots Using Discriminative Dimensionality Reduction. IJCAI 2020.* (open access). Also available on [Arxiv](https://arxiv.org/abs/1909.09154).

## Requirements

All requirements to run DeepView are in ```requirements.txt```. 
Most of the deep netowrks in the notebooks are models implemented in ```PyTorch```/```torchvision``` but we have also an example for ```Tensorflow 2``` models. We ran the Demo with versions ```torch==1.3.1```, ```torchvision==0.4.2```, ```tensorflow==2.0.0``` or for gpu support ```tensorflow-gpu==2.0.0```.

## Installation

 1. Download the repository
 2. Inside the DeepView-directory, run ```pip install -e .```<br>or copy the ```deepview``` folder into your working directory

## Usage Instructions

> For detailed usage, look at the demo notebook ```DeepView Demo.ipynb``` and ```demo.py```
 
 1. To enable deepview to call the classifier, a wrapper-function (like ```pred_wrapper``` in the Demo notebook) must be provided. DeepView will pass a numpy array of samples to this function and as a return, it expects according class probabilities (valid probability distribution, not the logits) from the classifier as numpy arrays.
 2. Initialize DeepView-object and pass the created method to the constructor
 3. Run your code and call ```add_samples(samples, labels)``` at any time to add samples to the visualization together with the ground truth labels.
    * The ground truth labels will be visualized along with the predicted labels
    * The object will keep track of a maximum number of samples specified by ```max_samples``` and it will throw away the oldest samples first
 4. Call the ```show``` method to render the plot
    * When ```interactive``` is True, this method is non-blocking to allow plot updates.
    * When ```interactive``` is False, this method is blocking to prevent termination of python scripts.
    

### Basic parameters

The following parameters control the basic functionality. We mark the absolutely required parameters with an exclamation mark (!) at the beginning.
(Parameters controlling the behaviour of DeepView are listed in the subsection below.)


| Variable               | Meaning           |
|------------------------|-------------------|
| (!)```pred_wrapper```     | Wrapper function allowing DeepView to use your model. Expects a single argument, which should be a batch of samples to classify. Returns (valid / softmaxed) prediction probabilities for this batch of samples. |
| (!)```classes```          | Names of all different classes in the data. |
| (!)```max_samples```      | The maximum amount of samples that DeepView will keep track of. When more samples are added, the oldest samples are removed from DeepView. |
| (!)```batch_size```       | The batch size used for classification |
| (!)```data_shape```       | Shape of the input data (complete shape; excluding the batch dimension) |
| ```resolution```       | x- and y- Resolution of the decision boundary plot. A high resolution will compute significantly longer than a lower resolution, as every point must be classified, default 100. |
| ```cmap```             | Name of the colormap that should be used in the plots, default 'tab10'. |
| ```interactive```      | When ```interactive``` is True, this method is non-blocking to allow plot updates. When ```interactive``` is False, this method is blocking to prevent termination of python scripts, default True. |
| ```title```            | Title of the deepview-plot. |
| ```data_viz```         | DeepView has a reactive plot, that responds to mouse clicks and shows the according data sample, when it is clicked. You can pass a custom visualization function, if ```data_viz``` is None, DeepView will try to show each sample as an image, if possible. (optional, default None)  |
| ```mapper```           | An object that maps samples from the data space to 2D space. Normally UMAP is used for this, but you can pass a custom mapper as well. (optional)  |
| ```inv_mapper```       | An object that maps samples from the 2D space to the data space. Normally ```deepview.embeddings.InvMapper``` is used for this, but you can pass a custom inverse mapper as well. (optional)  |
| ```kwargs```       | Configuration for the embeddings in case they are not specifically given in mapper and inv_mapper. Defaults to ```deepview.config.py```.  (optional)  |


### Parameters that influence the resulting visualization

DeepView essentially consists of three parts: an estimation of the Fisher metric, dimensionality reduction with UMAP and an inverse mapping. These are computed in this order, and hence the parameters of the Fisher metric influence both projections. The following lists the parameters of DeepView sortet by their importance (the first is the most important one):

|Variable            | Meaning          |
|--------------------|------------------|
| ```lam```          | (Fisher metric parameter) Controls the amount of euclidean regularization of the Fisher metric, the larger the more. Between 0 and 1, default 0.65. |
| ```n_neighbors```  | (UMAP parameter) Number of neighbors used in UMAP. Determines how many points are considered to be close for each data point, roughly speaking. Default is 30. |
| ```a```            | (inverse mapping parameter) Determines the nonlinearity of the inverse mapping. Large values correspond to nonlinear functions. Compared to the definition in the paper, we additionally devide by the range of the data, such that the letter can be ignored in setting this parameter. Default is 500. 
| ```min_dist```     | (UMAP parameter) The minimum distance between embedded points. Smaller values cause more clustered or clumped visualizations. Interdepends with ```spread```. Default is 0.1. |
| ```spread```       | (UMAP parameter) The scale of embedded points. Together with ```min_dist``` causes more/less clustering. Default is 1.0.|
|```n```             | (Fisher metric parameter) Number of interpolation steps for distance calculation between two points. In the paper, this is also called n, default 5. |
| ```random_state``` | (UMAP parameter) Seed used by UMAP. |

The UMAP and inverse mapping parameters are set with the ```_init_mappers``` function (see ```DeepView Demo_FashionMnist_BackdoorAttack``` for an example).

## The λ-Parameter

The λ-Hyperparameter (```lam```) weights the euclidian distance component against the discriminative (Jensen-Shannon) distance component:

<center>
<img alt="distance_equation" src="https://user-images.githubusercontent.com/30961397/84257875-557b5e00-ab16-11ea-9e75-fe1d7795739b.png" width=400px>
</center>

When the visualization doesn't show class-clusters, **try a smaller lambda** to put more emphasis on the discriminative (JS) distance component, which considers the class-predictions. A smaller λ will normally pull the datapoints further into their class-clusters. Therefore, **if λ is too small**, this can lead to collapsed clusters that don't represent any structural properties of the datapoints. Of course this behaviour also depends on the data and how well the label corresponds to certain structural properties.


## Sample visualization

![visualization](https://user-images.githubusercontent.com/30961397/71091913-628e4480-21a6-11ea-8a26-d94f13907548.png)
