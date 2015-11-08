require 'nn'

-- convolution neural network model
function cnn_model(classes)
	local model = nn.Sequential()
	model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
	model:add(nn.Tanh())
	model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
	-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
	model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
	model:add(nn.Tanh())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- stage 3 : standard 2-layer MLP:
	model:add(nn.Reshape(64*2*2))
	model:add(nn.Linear(64*2*2, 200))
	model:add(nn.Tanh())
	model:add(nn.Linear(200, #classes))

	model:add(nn.LogSoftMax())
	return model
end
