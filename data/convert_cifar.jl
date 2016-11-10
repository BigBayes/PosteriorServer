using MAT, HDF5

#cifar-10 with batches

# Read in the data from the Matlab file
# NOTE: Change this line if your file is elsewhere

# download cifar 10 dataset
run(`wget https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz`)

# extract cifar 10 dataset
run(`tar -zxvf cifar-10-matlab.tar.gz`)

# Specify data location
cifar_10_path = realpath("./cifar-10-batches-mat/")
# read in batches
train1 = matread("$(cifar_10_path)/data_batch_1.mat")
train2 = matread("$(cifar_10_path)/data_batch_2.mat")
train3 = matread("$(cifar_10_path)/data_batch_3.mat")
train4 = matread("$(cifar_10_path)/data_batch_4.mat")
train5 = matread("$(cifar_10_path)/data_batch_5.mat")
vars_test = matread("$(cifar_10_path)/test_batch.mat")

#concatenate data
training_images = [train1["data"];train2["data"];train3["data"];train4["data"];train5["data"]]
training_labels = [train1["labels"];train2["labels"];train3["labels"];train4["labels"];train5["labels"]]
test_images = vars_test["data"]
test_labels = vars_test["labels"]

# Calculate the variable sizes
count_channels = 3
count_training_samples = size(training_images, 1)
count_test_samples = size(test_images, 1)
width_image = convert(Int64, sqrt(size(training_images, 2) / count_channels))
height_image = width_image
size_image = width_image * height_image

# Preallocate the arrays
training_images_f = zeros(Float64,count_training_samples,height_image*width_image*count_channels)
test_images_f = zeros(Float64,count_test_samples,height_image*width_image*count_channels)
testx = zeros(Float64, height_image, width_image, count_channels, count_test_samples)
testc = zeros(Float64, 1, count_test_samples)

trainx = zeros(Float64, height_image, width_image, count_channels, count_training_samples)
trainc = zeros(Float64, 1, count_training_samples)

# Shuffle the data
idx_training = randperm(count_training_samples)
idx_test = randperm(count_test_samples)


# global contrast normalisation
# subtract mean across features per example
# and normalise by standard dev across features per example

for i in 1:count_training_samples
	training_images_f[i,:] = training_images[i,:]
	training_images_f[i,:] = (training_images_f[i,:] - mean(training_images_f[i,:]))/std(training_images_f[i,:])
end

for i in 1:count_test_samples
	test_images_f[i,:] = test_images[i,:]
	test_images_f[i,:] = (test_images_f[i,:] - mean(test_images_f[i,:]))/std(test_images_f[i,:])
end



# ZCA whitening (see Alex Krizhevsky's technical report)
# images are in n by d array

#find covariance
cov = training_images_f' *training_images_f

#find eigenfactorisation
eigen = eigfact(cov) # could use svd instead

# calculate whitening matrix
W = eigen[:vectors]*diagm(1./sqrt(1e-5+eigen[:values]))*eigen[:vectors]'
#
training_images_f = training_images_f*W'
test_images_f = test_images_f*W'

# check covariance is diagonal
# training_images_f'*training_images_f

# Convert to the format used in Mocha
for i in 1:count_training_samples
	class = training_labels[idx_training[i], 1]
	trainc[1, i] = class
	for j = 1:count_channels
		trainx[:, :, j, i] = reshape(training_images_f[idx_training[i], ((j - 1)*size_image + 1):(j * size_image)], height_image, width_image)'
		#println("$((j - 1)*size_image + 1)-$(j * size_image)")
	end
end

for i in 1:count_test_samples
	class = test_labels[idx_test[i], 1]
	testc[1, i] = class
	for j = 1:count_channels
		testx[:, :, j, i] = reshape(test_images_f[idx_test[i], ((j - 1)*size_image + 1):(j * size_image)], height_image, width_image)'
	end
end


# Save to a .hdf5 file
file_train = h5open("cifar_10_gcn_zca_train.hdf5", "w")
file_train["data"] = trainx
file_train["label"] = trainc
close(file_train)

file_test = h5open("cifar_10_gcn_zca_test.hdf5", "w")
file_test["data"] = testx
file_test["label"] = testc
close(file_test)
