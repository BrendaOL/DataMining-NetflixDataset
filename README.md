# DataMining-NetflixDataset
Estimation of ratings for pairs (movie, user) that do not appear in the Netflix dataset for a recommender system. In essence, we implemented both SGD and BGD to iteratively refine the latent factor matrices P and Q to improve the accuracy of predicting ratings for user-movie pairs that don't have explicit ratings in the original dataset. They serve as optimization techniques to minimize the prediction error and enhance the recommendation system's performance.

It was developed in python using Jupiter notebook with the libraries NumPy and SciPy for creating and manipulating sparse matrices, as well as Matplotlib for plotting.
### _This project is part of the Information Retrieval and Data Mining (2022) course at the Vrije Universiteit Brussels under the supervision of Prof. Pieter Libin - Dr. Denis Steckelmacher._
---
### The Netflix Challenge dataset
Because of the size of the dataset, the file cannot be shared in GitHub; however this file is public and information about it can be found in the DataInformation file.
The following steps illustrate how we overcome the issues of Netflix file structure, dealing with the size of the lists to create the sparse matrix:
- Temporary files were created to store movies, users, and ratings independently.
- A function readLines() was developed to read Netflix files and save the data in the temporary files, this way the Netflix’s file is only iterated once.
- Using the data from the files, a sparse matrix was built, and the matrix was saved in a separate file to be used for the following tasks.
- The temporary files are no longer needed because the matrix will be used exclusively from now on, so the files were therefore deleted.
This solution was achieved through trial and error; each iteration of each file (four Netflix files) took between 6 and 8 minutes, for a total of around 32 minutes to create the sparse matrix; despite the long duration, no improvements were made because it was a one-time process

### SGD and BGD
For the implementation of the SGD, several readings were made to understand the details of the algorithm, among them was reviewed the article Netflix challenge[^1], in this article was found Simon Funk, "popularized a stochastic gradient descent optimization"[^2] which allowed a better understanding of the operations to be performed and whose operations are used in the implementation. Finally, we read the Nicolas Hug, "Understanding matrix factorization for recommendation"[^3] blog which allowed us to understand the SGD procedure and which was also used in the development of the SGD. The initialization code on this project is inspired by the same way of Nicolas Hug’s initialization of P and Q. 

[^1]: https://pantelis.github.io/cs634/docs/common/lectures/recommenders/netflix/
[^2]: https://sifter.org/~simon/journal/20061211.html
[^3]: http://nicolas-hug.com/blog/matrix_facto_3
