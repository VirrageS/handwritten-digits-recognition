require 'nn'
require 'optim'
require './load_dataset'
require './cnn_model.lua'

torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

op = xlua.OptionParser('process_videos.lua [options]')
op:option{'-train', '--traing_data_size', action='store', dest='trainDataSize', help='directory to load videos', default=5000}
op:option{'-test', '--test_data_size', action='store', dest='testDataSize', help='only load files of this extension', default=1000}
op:option{'-lr', '--learning_rate', action='store', dest='learningRate', help='folder to output files', default=0.05}
op:option{'-b', '--batch_size', action='store',dest='batchSize', help='number of frames per batch', default=10}
opt = op:parse()
opt.batchSize = tonumber(opt.batchSize)
opt.learningRate = tonumber(opt.learningRate)

-- load datasets
trainData = loadDataset('data/train_32x32.t7')
testData = loadDataset('data/test_32x32.t7')

-- this matrix records the current confusion across classes
classes = {'1','2','3','4','5','6','7','8','9','10'}
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger('train.log')
testLogger = optim.Logger('test.log')

-- load model
model = cnn_model(classes)

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
			local sample = dataset[i]
			local input = sample[1]:clone()
			local _,target = sample[2]:clone():max(1)

			target = target:squeeze()
			inputs[k] = input
			targets[k] = target
			k = k + 1
		end

		local feval = function(x)
			-- get new parameters
			if x ~= parameters then
				parameters:copy(x)
			end

			-- reset gradients
			gradParameters:zero()

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

		-- Perform SGD step:
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
	for t = 1, opt.testDataSize do
		local sample = dataset[t]
		local input = sample[1]:clone()
		local _, target = sample[2]:clone():max(1)
		target = target:squeeze()

		local predicted = model:forward(input)
		confusion:add(predicted, target)

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
