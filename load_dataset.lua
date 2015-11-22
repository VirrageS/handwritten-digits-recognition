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
		os.execute([['
			wget ' .. remote .. '; mkdir data;
			tar xvf ' .. tar .. '; rm ' .. tar .. ';
			sudo mv -v mnist.t7/* data/; sudo rm -rf mnist.t7
		']])
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

------------------
-- [[ KAGGLE ]] --
------------------

function downloadKaggleDataset(file)
	if not paths.filep(file) then
		local remoteTest = 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/3004/test.csv?sv=2012-02-12&se=2015-11-25T11%3A50%3A34Z&sr=b&sp=r&sig=BL8xNMsxgwP%2BBVVwp44jkIi%2FHij1HZrBzTkvOMY5oHs%3D'
		local remoteTrain = 'https://kaggle2.blob.core.windows.net/competitions-data/kaggle/3004/train.csv?sv=2012-02-12&se=2015-11-25T11%3A50%3A04Z&sr=b&sp=r&sig=zIpC6o2bzAuijY6XYrMYv%2FtU%2FSt%2FOzPh%2B8A44V7Fk%2BM%3D'

		os.execute('mkdir data;')
		os.execute('wget -O test.csv "' .. remoteTest .. '"; mv test.csv data/')
		os.execute('wget -O train.csv "' .. remoteTrain .. '"; mv train.csv data/')
	end
end

function loadKaggleTrainDataset()
	print('LOADING KAGGLE TRAIN DATA')

	local file = "data/train.csv"
	downloadKaggleDataset(file)

	local lines = {}
	for line in io.lines(file) do
		table.insert(lines, line)
	end
	table.remove(lines, 1) -- remove headers

	local dataset = {}
	for idx = 1, #lines do
		local pixels = lines[idx]:split(",")
		local label = torch.Tensor(1):zero()
		local image = torch.Tensor(28, 28):zero()

		for row = 1, 28 do
			for column = 1, 28 do
				image[row][column] = tonumber(pixels[(row - 1) * 28 + column + 1])
			end
		end

		label[1] = pixels[1] + 1
		table.insert(dataset, {[1] = image, [2] = label})
		xlua.progress(idx, #lines)
	end

	return dataset
end

function loadKaggleTestDataset()
	print('LOADING KAGGLE TEST DATA')

	local file = "data/test.csv"
	downloadKaggleDataset(file)

	local lines = {}
	for line in io.lines(file) do
		table.insert(lines, line)
	end
	table.remove(lines, 1) -- remove headers

	local dataset = {}
	for idx = 1, #lines do
		local pixels = lines[idx]:split(",")
		local image = torch.Tensor(28, 28):zero()

		for row = 1, 28 do
			for column = 1, 28 do
				image[row][column] = tonumber(pixels[(row - 1) * 28 + column])
			end
		end

		table.insert(dataset, {[1] = image})
		xlua.progress(idx, #lines)
	end

	return dataset
end
