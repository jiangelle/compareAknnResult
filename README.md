The experiment is performed under the http://corpus-texmex.irisa.fr SIFT1M dataset.If you want to perform the GIFT1M dataset, you should change the isSift from true to false
in kdtreeexperiment.cpp.
You can directly debug the experiment in kdtree project.
The groundtruth result is the data we read from the sift_groundtruth.ivecs.And the size of the data is query*knn .It records the indices of the features that are
the knn neighbors of the query.
The kdtree parameters are adjusted to get close to the groundtruth,we found when the tree is 1,the result is best.
The FLSH parameters are table counts (30),bucket counts for each table (2000) , and the mul-prob that is default to be zero.