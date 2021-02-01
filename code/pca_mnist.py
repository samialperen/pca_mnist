import numpy as np
from mlxtend.data import loadlocal_mnist #to convert mnist dataset to np arrays
from pathlib import Path #deal with local and global paths
from matplotlib import pyplot as plt 

# If you run the code from pattern_recognition_assignment1 path
data_path = Path.cwd() / 'data' 
# If you run the code from code directory
#data_path = Path.cwd() / ".." / "data"/
image_files_path =  str(data_path / 'train-images.idx3-ubyte')
label_files_path =  str(data_path / 'train-labels.idx1-ubyte')

XT, true_labelsT = loadlocal_mnist(
        images_path= image_files_path, 
        labels_path= label_files_path)

X = XT.T # X=DxN=784x6000 #Data matrix
true_labels = true_labelsT.T # labels=1x6000 

# It is required for reconstruction
mu= np.mean(X,axis=1,keepdims=True) #mu= Dx1

################ Implementation of PCA
def f_PCA(input_data,d):
        # input_data = matrix DxN where D=# dimensions and N=# of samples
        # d = It decides output dimension --> output=dxN
        # output: input_data but dimension is d --> so output size dxN
        # output 2: W matrix --> which is useful for inverse PCA 
        # Step 1
        mu= np.mean(input_data,axis=1,keepdims=True) #mu= Dx1
        cov= np.cov(input_data) #cov= DxD

        # Step 2
        input_data_hat = input_data- mu #input_data_hat has zero mean --> shape DxN

        # Step 3 
        Evalues, Evectors= np.linalg.eig(cov)
        
        # Step 4  
        # Select the largest d eigenvalues 
        # It is sorted, so principal_eigenvalues[0] >  principal_eigenvalues[1] > ... > 
        p_evalues_col_indices = Evalues.argsort()[::-1][:d]
        W = Evectors[:,p_evalues_col_indices]

        # Step 5
        y = np.dot(W.T,input_data_hat) #matrix multiplication
        return y, W

################ Selecting a Suitable Dimension d Corresponding to Proportion of Variance (POV) 95\%
cov = np.cov(X) #cov= DxD
eigenvalues, _ = np.linalg.eig(cov)
total_of_all_eigenvalues = np.sum(eigenvalues)
desired_pov = 95/100 #percent
eigenvalues_sorted = -np.sort(-eigenvalues) #ascending order, i.e. first elements is the largest

calculated_pov = 0
epsilon = 0.001
# calculated_pov - desired_pov might be not exactly same, it is good practice to
# put some epsilon value
eigenvalue_sum = 0 
d = 0 #counter for while loop, also our d value for PCA algorithm 
while abs(calculated_pov - desired_pov) > epsilon:
        eigenvalue_sum += eigenvalues_sorted[d]
        calculated_pov = eigenvalue_sum / total_of_all_eigenvalues
        print("Desired pov is ", desired_pov)
        print("Calculated pov is ", calculated_pov)
        d += 1

# Since python indices start from zero, we need to add 1 after the loop
# to get the real d value
d += 1 
print("d value corresponding to ", desired_pov, "POV is ", d)

################ Selecting a Suitable Dimension d Considering Average Mean Square Error (MSE)

# First construct inverse PCA function (PCA Reconstruction)
def f_inv_PCA(input_data,W):
        # input_data = matrix dxN where d=# reduced dimensions and N=# of samples
        # W = size dxD matrix containing all eigenvectors of original data 
        # output: original data size dxN, but with zero mean

        reconstructed_data_hat = np.dot(W,input_data)  
        return reconstructed_data_hat  

# Calculate MSE for the d_values 
d_values = np.append(np.insert(np.arange(20,780,20),0,1),784)
MSE_outputs = np.zeros((d_values.shape[0],1))

for i in range(d_values.shape[0]):
        reduced_d, W = f_PCA(X,d_values[i])
        reconstructed_d = f_inv_PCA(reduced_d,W) + mu 
        error_square_d = (X-reconstructed_d)**2 #matrix DxN
        features_means_for_each_sample_d = np.mean(error_square_d,axis=0) #vector 1xN
        MSE_outputs[i] = np.mean(features_means_for_each_sample_d) #a scalar


fig1 = plt.figure()
plt.plot(d_values,MSE_outputs)
plt.title("Mean Square Error (MSE) variation with d")
plt.xlabel("d")
plt.ylabel("MSE")


################ Reconstruction of Original Image
number8_data = X[:,true_labels==8] #this contains all data belong to class of number 8 
one_sample_number8 = number8_data[0] #just one example of data belong to class of number 8
# one_sample_number8 size --> 784x1 


d_values_number8 = np.array([1,10,50,250,784])
# This reconstructed dictionary holds reconstructed image 8 for different d values
# For ex,reconstructed_sample_class8[1] is the reconstructed image 8 for d=10
# I am selecting the first image whose class is 8 with:
# reconstructed_sample[:,true_labels==8][:,0]
reconstructed_sample_class8 = {} 
for i in range(d_values_number8.shape[0]):
        #we still need to use all data to train PCA
        reduced_samples, W = f_PCA(X,d_values_number8[i]) 
        reconstructed_sample = f_inv_PCA(reduced_samples,W) + mu
        reconstructed_sample_class8[i] = reconstructed_sample[:,true_labels==8][:,0]



# Show original image first 
original_image_class8 = X[:,true_labels==8][:,0].reshape((28,28))
fig2 = plt.figure()
plt.title("Original Sample from the Class of Number 8")
plt.imshow(original_image_class8)


# Now show reconstructed ones
for i in range(d_values_number8.shape[0]):
        reconstructed_array = reconstructed_sample_class8[i].reshape((28,28))
        plt.figure()
        plt.title("Reconstructed Sample with d=%d" %d_values_number8[i])
        plt.imshow(reconstructed_array)
        



################ Analyze Variation of Eigenvalues with Reduced Dimension d
# It ask for to display sorted eigenvalues
fig3 = plt.figure()
plt.plot(eigenvalues_sorted)
plt.title("Eigenvalues varying with d value")
plt.xlabel("d")
plt.ylabel("Eigenvalue")


print("Q3 is done. Close the figures to see results of other question")
plt.show()




