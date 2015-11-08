require 'torch'
require 'paths'

function loadTrainDataset()
	return loadDataset('data/train_32x32.t7')
end

function loadTestDataset()
	return loadDataset('data/test_32x32.t7')
end

function downloadDataset(file)
	if not paths.filep(file) then
		local remote = 'https://s3.amazonaws.com/torch7/data/mnist.t7.tgz'
		local tar = paths.basename(remote)
		os.execute('wget ' .. remote .. '; mkdir data; tar xvf ' .. tar .. '; rm ' .. tar .. '; sudo mv -v mnist.t7/* data/; sudo rm -rf mnist.t7')
	end
end


function loadDataset(file)
	downloadDataset(file)

	local f = torch.load(file, 'ascii')
	local data = f.data:type(torch.getdefaulttensortype())
	local labels = f.labels

	local dataset = {}
	for i = 1, f.data:size(1) do
		label = torch.Tensor(1)
		label[1] = labels[i]
		table.insert(dataset, {[1] = data[i], [2] = label})
	end

	return dataset
end
