# Similar images finder (using transfer learning)

Given a set of database images, we take the trained image classification VGG network, remove its last layers, and use the dissected model to convert our raw images into feature vectors for similarity comparison to produce similar image recommendations. No training is needed as we are re-using the low-level weight layers in the VGG network. A schematic for our implementation is shown here:


The steps towards building our similar images finder:

1. Prepare our image database. We prepared by default a 36 images database of common steakhouse foods (6 classes).

2. Take the VGG model and remove its last layers.

3. Convert our image database into feature vectors using our dissected VGG model. If the output layer of the dissected model are convolutional filters then flatten the filters and append them make a single vector.

4. Compute similarities between our image feature vectors using an inner-product such as cosine similarity or euclidean distance

5. For each image, select the top-k images that have the highest similarity scores to build the recommendation


### Usage:

1. Place your database of images into the `db` directory.

2. Run the command:

    > python similar_images_TL.py 

    All output from running this code will be placed in the `output` directory. There will be a `tsne.png` plot for the t-SNE visualization of your database image embeddings, as well as a `rec` directory containing the top `k = 5` similar image recommendations for each image in your database.

If the program is running properly, you should see something of the form:

```
Loading VGG19 pre-trained model...
Reading images from 'db' directory...

imgs.shape = (39, 224, 224, 3)
X_features.shape = (39, 100352)

### Required libraries:

* keras, numpy, matplotlib, sklearn, h5py, pillow

### Authors:

Anson Wong
