require 'nn'

function linear_model(classes)
	local model = nn.Sequential()
	model:add(nn.View(32 * 32))
	model:add(nn.Linear(32 * 32, 1000))
	model:add(nn.ReLU())

	model:add(nn.Linear(1000, 1000))
	model:add(nn.BatchNormalization(1000))
	model:add(nn.ReLU())
	model:add(nn.Dropout(0.5))

	model:add(nn.Linear(1000, #classes))
	model:add(nn.LogSoftMax())
	return model
end
