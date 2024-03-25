import numpy as np
import matplotlib.pyplot as plt

def distance(data, centers):
    '''
    Calculate the distances from points to cluster centers.
      parameter:
        data: nparray of shape (n, 2)
        centers: nparray of shape (m, 2)
      return:
        distance: nparray of shape (n, m)
    '''
    #get the m,and n from data and centers
    n, _ = data.shape
    m, _ = centers.shape
    #initialize a nparray
    dist = np.zeros((n,m))
    for j in range(m):
      #each row is difference between a data point at the j-th center.
      b = data - centers[j]
      #calculates the square root of the sum of squares of components of these vectors along row-wise
      #calculate the Euclidean distance of each data point from the j-th center
      dist[:, j] = np.linalg.norm(b, axis=1)
    return dist

def kmeans(data, n_centers):
    """
    Divide data into n_centers clusters, return the cluster centers and assignments.
    Assignments are the indices of the closest cluster center for the corresponding data point.
      parameter:
        data: nparray of shape (n, 2)
        n_centers: int
      return:
        centers: nparray of shape (n_centers, 2)
        assignments: nparray of shape (n,)
    """
    
    n, _ = data.shape
    assignments = np.zeros(n, dtype=int)
    
    #randomly selecting data points
    random_indices = np.random.choice(n, n_centers, replace=False)
    centers = data[random_indices]
    check = False
    while not check:
        #Use the distance function to Calculate distance from each data point to each center
        dist = distance(data, centers)
        
        #Assign data points to the nearest center
        nearest = np.argmin(dist, axis=1)
        
        # if they are optimal, stop
        if np.array_equal(nearest, assignments):
            check = True
        assignments = nearest
        
        # Calculate new centers as the mean of points assigned to each center
        new_centers = np.zeros_like(centers)
        for k in range(n_centers):
            if np.any(assignments == k):
                #reference:https://analyticsarora.com/k-means-for-beginners-how-to-build-from-scratch-in-python/
                new_centers[k] = np.mean(data[assignments == k], axis=0)
            else:
                #randomly seelct the data as center, if no points are assigned to it
                random_data = np.random.choice(n, 1)
                new_centers[k] = data[random_data]
        
        # If centers haven't changed , stop
        if np.allclose(new_centers, centers):
            check = True
        else:
          centers = new_centers

    return centers, assignments



def distortion(data, centers, assignments):
    """
    Calculate the distortion of the clustering.
      parameter:
        data: nparray of shape (n, 2)
        centers: nparray of shape (m, 2)
        assignments: nparray of shape (n,)
      return:
        distortion: float
    """
    total_distortion = 0
    for i in range(len(data)):
        point = data[i]
        center_idx = assignments[i]
        
       
        center = centers[center_idx]
        
       
        squared_distance = np.sum((point - center) ** 2)
        
       
        total_distortion += squared_distance
    
    return total_distortion

def spectral_clustering(data, n_centers, sigma):
    """
    Divide data into n_centers clusters, return the assignments.
    Assignments are the indices of the cluster center for the corresponding data point.
      parameter:
        data: nparray of shape (n, 2)
        n_centers: int
        sigma: float
      return:
        assignments: nparray of shape (n,)
    """
    #reference:https://medium.com/@roiyeho/spectral-clustering-50aee862d300
    #reference:https://towardsdatascience.com/spectral-graph-clustering-and-optimal-number-of-clusters-estimation-32704189afbe
    #step1:
    n = data.shape[0]
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                A[i, j] = np.exp(-np.sum((data[i] - data[j])**2) / (2 * sigma**2))
    
    # Step 2: Define D and construct L,use gpt do this step
    #D is a diagonal matrix where each diagonal element,Dii is is the sum of the i-th  row of the affinity matrix A
    D = np.diag(np.sum(A, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    #The Laplacian matrix L
    L = D_inv_sqrt @ A @ D_inv_sqrt
    
    # Step 3: Find k largest eigenvectors of L
    #Find the k largest eigenvectors of L, ensuring they are orthogonal to each other in case of repeated eigenvalues. 
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx_k_largest = eigenvalues.argsort()[::-1][:n_centers]
    X = eigenvectors[:, idx_k_largest]
    
    # Step 4: Normalize the largest eigenvectors
    Y = X / np.sqrt(np.sum(X**2, axis=1, keepdims=True))
    
    # Step 5: Apply kmeans on rows of Y = X_norm
    _, assignments = kmeans(Y, n_centers)
    
    return assignments
    

def question2(data):
    #distortions
    distortions = []
    #print(data.shape)

    temp = []
  
    for i in range(1,11):
        index = 10 * i
        centers, assignments = kmeans(data, index)
        dist = distortion(data, centers, assignments)
        distortions.append(dist)
        temp.append(index)
    
    for i in range(1,8):
        index = 100 * i
        centers, assignments = kmeans(data, index)
        dist = distortion(data, centers, assignments)
        distortions.append(dist)
        temp.append(index)
    centers, assignments = kmeans(data, 788)
    dist = distortion(data, centers, assignments)
    distortions.append(dist)
    print(distortions)
    temp.append(788)
    
    #plot question2
    plt.plot(temp, distortions, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title('Distortion vs. Number of Clusters')
    plt.savefig('Distortion.png')
    plt.show()
    #print(temp)
if __name__ == "__main__":
    ''' 
    main function here
    Run kmeans and spectral clustering on the 2d data given. 
    Use matplotlib to visualize the clustering results of the two methods.
    Save them as png files.
    '''

    data = np.load('./data_Aggregation.npy')
    n_centers = 7
    sigma = 0.72
    #question2 code:
    #question2(data)
    #kmeans plot
    centers, assignments = kmeans(data, n_centers)
    plt.scatter(data[:, 0], data[:, 1], c=assignments, cmap='viridis', alpha=0.5, marker='.')# gpt
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.9, marker='x')  # gpt
    plt.title(f'K-means Clustering with {n_centers} Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    # Save the plot
    plt.savefig(f'kmeans_clustering_{n_centers}_clusters.png')
    plt.show()
    #spectral_clustering plot
    assignments = spectral_clustering(data, n_centers, sigma)

    # Plot the results
    plt.scatter(data[:, 0], data[:, 1], c=assignments, cmap='viridis', alpha=0.5, marker='.')# gpt
    plt.title(f'Spectral Clustering with {n_centers} Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(f'spectral_clustering')
    plt.show()

   

    
    