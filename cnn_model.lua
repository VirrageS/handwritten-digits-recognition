require 'nn'

-- convolution neural network model
function cnn_model(classes) -- 99,22% on best iteration
	local model = nn.Sequential()
	model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

	model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

	-- fully connected
	model:add(nn.View(64 * 5 * 5))
	model:add(nn.Linear(64 * 5 * 5, 1000))
	model:add(nn.ReLU())
	model:add(nn.Dropout(0.5))

	model:add(nn.Linear(1000, 1000))
	model:add(nn.BatchNormalization(1000))
	model:add(nn.ReLU())
	model:add(nn.Dropout(0.5))

	model:add(nn.Linear(1000, #classes))
	model:add(nn.LogSoftMax())
	return model
end

-- convolution neural network model
function cnn_kaggle_model(classes) -- 99,22% on best iteration
	local model = nn.Sequential()
	model:add(nn.SpatialConvolutionMM(1, 20, 5, 5))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

	model:add(nn.SpatialConvolutionMM(20, 40, 5, 5))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

	-- fully connected
	model:add(nn.View(40 * 4 * 4))
	model:add(nn.Linear(40 * 4 * 4, 1000))
	model:add(nn.ReLU())
	model:add(nn.Dropout(0.5))

	model:add(nn.Linear(1000, 1000))
	model:add(nn.BatchNormalization(1000))
	model:add(nn.ReLU())
	model:add(nn.Dropout(0.5))

	model:add(nn.Linear(1000, #classes))
	model:add(nn.LogSoftMax())
	return model
end
