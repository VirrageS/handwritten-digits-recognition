require 'torch'

function loadDataset(file)
	local f = torch.load(file, 'ascii')
	local data = f.data:type(torch.getdefaulttensortype())
	local labels = f.labels

	local dataset = {}
	for i = 1, f.data:size(1) do
		local labelvector = torch.zeros(10)
		local label = labelvector:zero()
		label[labels[i]] = 1

		table.insert(dataset, {[1] = data[i], [2] = label})
	end

	return dataset
end
