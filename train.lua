require 'nn'
require 'optim'
require './load_dataset'
require './cnn_model'
require './linear_model'

op = xlua.OptionParser('train.lua [options]')
op:option{'-train_size', '--traing_data_size', action='store', dest='trainDataSize', help='size of training sets', default=60000}
op:option{'-test_size', '--test_data_size', action='store', dest='testDataSize', help='size of testing sets', default=10000}
op:option{'-rate', '--learning_rate', action='store', dest='learningRate', help='learning rate', default=0.05}
op:option{'-batch', '--batch_size', action='store', dest='batchSize', help='number of sets in batch', default=10}
op:option{'-t', '--threads', action='store', dest='threads', help='number of threads used by networks', default=2}
op:option{'-seed', '--seed', action='store', dest='seed', help='seed', default=130}
op:option{'-gpuid', '--enable_gpu', action='store', dest='gpuid', help='loads gpu (needs cunn and cutorch)', default=-1}

opt = op:parse()
opt.batchSize = tonumber(opt.batchSize)
opt.learningRate = tonumber(opt.learningRate)
opt.threads = tonumber(opt.threads)
opt.gpuid = tonumber(opt.gpuid)
opt.seed = tonumber(opt.seed)

torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

-- load GPU
if opt.gpuid >= 0 then
	local ok, cunn = pcall(require, 'cunn')
	local ok2, cutorch = pcall(require, 'cutorch')
	if not ok then print('package cunn not found!') end
	if not ok2 then print('package cutorch not found!') end

	if ok and ok2 then
		print('using CUDA on GPU ' .. opt.gpuid .. '...')
		cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
		cutorch.manualSeed(opt.seed)
	else
		print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
		print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
		print('Falling back on CPU mode')
		opt.gpuid = -1 -- overwrite user setting
	end
end

-- load datasets
trainData = loadTrainDataset()
testData = loadTestDataset()

classes = {'1','2','3','4','5','6','7','8','9','10'} -- classification classes
-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

model = cnn_model(classes) -- load model
criterion = nn.ClassNLLCriterion() -- loss function
if opt.gpuid >= 0 then -- if cuda is enabled convert models to cuda models
	model = model:cuda()
	criterion = criterion:cuda()
end

-- retrieve parameters and gradients from model (cuda's or normal one)
parameters, gradParameters = model:getParameters()

-- SGD step:
sgdState = {
	learningRate = opt.learningRate,
	learningRateDecay = 1e-4,
	momentum = 0, weightDecay = 0
}

function train(dataset)
	local currentLoss = 0

	for t = 1, opt.trainDataSize, opt.batchSize do
		-- create mini batch
		local inputs = torch.Tensor(opt.batchSize, 1, 32, 32)
		local targets = torch.Tensor(opt.batchSize)

		local k = 1
		for i = t, math.min(t + opt.batchSize - 1, opt.trainDataSize) do
			inputs[k] = dataset[i][1]:clone() -- copy data
			targets[k] = dataset[i][2]:clone():squeeze() -- copy label
			k = k + 1
		end

		local feval = function(x)
			-- get new parameters
			if x ~= parameters then
				parameters:copy(x)
			end

			-- reset gradients
			gradParameters:zero()

			-- if we use cuda we must convert torchTensor to cudaTensor
			-- because our model requires it
			if opt.gpuid >= 0 then
				inputs = inputs:float():cuda()
				targets = targets:float():cuda()
			end

			-- evaluate the loss function and its derivative wrt x,
			-- for that sample
			local loss_x = criterion:forward(model:forward(inputs), targets)
			model:backward(inputs, criterion:backward(model.output, targets))

			for i = 1, opt.batchSize do
				confusion:add(model.output[i], targets[i])
			end

			return loss_x, gradParameters
		end

		_, fs = optim.sgd(feval, parameters, sgdState)
		currentLoss = currentLoss + fs[1]

		-- display progress
		xlua.progress(t, opt.trainDataSize)
	end

	print('\n\nCurrent loss: ' .. tostring(currentLoss / opt.trainDataSize) .. '\n')

	-- print confusion matrix
	print(confusion)
	confusion:zero()
end

function test(dataset)
	for t = 1, opt.testDataSize, opt.batchSize do
		-- create mini batch
		local inputs = torch.Tensor(opt.batchSize, 1, 32, 32)
		local targets = torch.Tensor(opt.batchSize)
		local k = 1
		for i = t, math.min(t + opt.batchSize - 1, opt.testDataSize) do
			inputs[k] = dataset[i][1]:clone() -- copy data
			targets[k] = dataset[i][2]:clone():squeeze() -- copy label
			k = k + 1
		end

		-- if we use cuda we must convert torchTensor to cudaTensor
		-- because our model requires it
		if opt.gpuid >= 0 then
			inputs = inputs:float():cuda()
			targets = targets:float():cuda()
		end

		-- predict
		local predicted = model:forward(inputs)

		for i = 1, opt.batchSize do
			confusion:add(predicted[i], targets[i])
		end

		-- dispaly progress
		xlua.progress(t, opt.testDataSize)
	end

	-- print confusion matrix
	print(confusion)
	confusion:zero()
end

while true do
	train(trainData)
	test(testData)
end
