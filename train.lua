require 'nn'
require 'optim'
require './load_dataset'
require './cnn_model'
require './linear_model'

function toboolean(X)
	return not not X
end

op = xlua.OptionParser('train.lua [options]')
op:option{'-train_size', '--traing_data_size', action='store', dest='trainDataSize', help='size of training sets', default=60000}
op:option{'-test_size', '--test_data_size', action='store', dest='testDataSize', help='size of testing sets', default=10000}
op:option{'-rate', '--learning_rate', action='store', dest='learningRate', help='learning rate', default=0.05}
op:option{'-batch', '--batch_size', action='store', dest='batchSize', help='number of sets in batch', default=10}
op:option{'-t', '--threads', action='store', dest='threads', help='number of threads used by networks', default=2}
op:option{'-seed', '--seed', action='store', dest='seed', help='seed', default=130}
op:option{'-gpuid', '--enable_gpu', action='store', dest='gpuid', help='loads gpu (needs cunn and cutorch)', default=-1}
op:option{'-kaggle', '--enable_kaggle', action='store', dest='kaggleEnabled', help='loads gpu (needs cunn and cutorch)', default=nil}

opt = op:parse()
opt.batchSize = tonumber(opt.batchSize)
opt.learningRate = tonumber(opt.learningRate)
opt.threads = tonumber(opt.threads)
opt.gpuid = tonumber(opt.gpuid)
opt.seed = tonumber(opt.seed)
opt.kaggleEnabled = toboolean(opt.kaggleEnabled)

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

if opt.kaggleEnabled then
	kaggleTrainData = loadKaggleTrainDataset()
	kaggleTestData = loadKaggleTestDataset()

	geometry = {28, 28}
	classes = {'0','1','2','3','4','5','6','7','8','9'} -- classification classes
else
	-- load datasets
	trainData = loadTrainDataset()
	testData = loadTestDataset()

	geometry = {32, 32}
	classes = {'1','2','3','4','5','6','7','8','9','10'} -- classification classes
end

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

if opt.kaggleEnabled then
	model = cnn_kaggle_model(classes) -- load model
else
	model = cnn_model(classes) -- load model
end

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
		local inputs = torch.Tensor(opt.batchSize, 1, geometry[1], geometry[2])
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

	if opt.kaggleEnabled then
		-- Opens a file in append mode
		os.execute('rm -f data/submission.csv; touch data/submission.csv')
		results = io.open('data/submission.csv', "a")
		results:write('"ImageId","Label"\n')
	end


	for t = 1, opt.testDataSize, opt.batchSize do
		-- create mini batch
		local inputs = torch.Tensor(opt.batchSize, 1, geometry[1], geometry[2])
		if not opt.kaggleEnabled then
			targets = torch.Tensor(opt.batchSize)
		end

		local k = 1
		for i = t, math.min(t + opt.batchSize - 1, opt.testDataSize) do
			inputs[k] = dataset[i][1]:clone() -- copy data

			if not opt.kaggleEnabled then
				targets[k] = dataset[i][2]:clone():squeeze() -- copy label
			end
			k = k + 1
		end

		-- if we use cuda we must convert torchTensor to cudaTensor
		-- because our model requires it
		if opt.gpuid >= 0 then
			inputs = inputs:float():cuda()

			if not opt.kaggleEnabled then
				targets = targets:float():cuda()
			end
		end

		-- predict
		local predicted = model:forward(inputs)

		if opt.kaggleEnabled then
			local _, prediction = predicted:max(1)
			prediction = prediction:transpose(1, 2)

			for i = 1, prediction:size(1) do
				results:write('' .. (t - 1 + i) .. ',"' .. tostring(prediction[i][1] - 1) .. '"\n')
			end
		else
			for i = 1, opt.batchSize do
				confusion:add(predicted[i], targets[i])
			end
		end

		-- dispaly progress
		xlua.progress(t, opt.testDataSize)
	end

	if opt.kaggleEnabled then
		-- closes the open file
		results:close()
	end

	if not opt.kaggleEnabled then
		-- print confusion matrix
		print(confusion)
		confusion:zero()
	end
end

while true do
	if opt.kaggleEnabled then
		train(kaggleTrainData)
		test(kaggleTestData)
	else
		train(trainData)
		test(testData)
	end
end
