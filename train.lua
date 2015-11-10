require 'nn'
require 'optim'
require './load_dataset'
require './cnn_model'
require './linear_model'

torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

op = xlua.OptionParser('train.lua [options]')
op:option{'-train_size', '--traing_data_size', action='store', dest='trainDataSize', help='size of training sets', default=60000}
op:option{'-test_size', '--test_data_size', action='store', dest='testDataSize', help='size of testing sets', default=10000}
op:option{'-rate', '--learning_rate', action='store', dest='learningRate', help='learning rate', default=0.05}
op:option{'-batch', '--batch_size', action='store', dest='batchSize', help='number of sets in batch', default=10}
op:option{'-t', '--threads', action='store', dest='threads', help='number of threads used by networks', default=2}

opt = op:parse()
opt.batchSize = tonumber(opt.batchSize)
opt.learningRate = tonumber(opt.learningRate)
opt.threads = tonumber(opt.threads)

torch.setnumthreads(opt.threads)

-- load datasets
trainData = loadTrainDataset()
testData = loadTestDataset()

-- classification classes
classes = {'1','2','3','4','5','6','7','8','9','10'}
-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- load model
model = linear_model(classes)

-- retrieve parameters and gradients from model
parameters, gradParameters = model:getParameters()

-- loss functions
criterion = nn.ClassNLLCriterion()


function train(dataset)
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

			-- compute outputs
			local outputs = model:forward(inputs)
			local f = criterion:forward(outputs, targets)

			-- estimate df/dW
			local df_do = criterion:backward(outputs, targets)
			model:backward(inputs, df_do)

			for i = 1, opt.batchSize do
				confusion:add(outputs[i], targets[i])
			end

			return f, gradParameters
		end

		-- SGD step:
		sgdState = sgdState or {
			learningRate = opt.learningRate,
			momentum = 0,
			learningRateDecay = 5e-7
		}
		optim.sgd(feval, parameters, sgdState)

		-- display progress
		xlua.progress(t, opt.trainDataSize)
	end

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
