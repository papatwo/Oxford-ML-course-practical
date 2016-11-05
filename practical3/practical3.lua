---------------------------------------------------------------------------------------
-- Practical 3 - Learning to use different optimizers with logistic regression
--
-- to run: th -i practical3.lua
-- or:     luajit -i practical3.lua
---------------------------------------------------------------------------------------

require 'torch'
require 'math'
require 'nn'
require 'optim'
require 'gnuplot'
require 'dataset-mnist'

------------------------------------------------------------------------------
-- INITIALIZATION AND DATA
------------------------------------------------------------------------------

torch.manualSeed(1)    -- fix random seed so program runs the same every time

-- TODO: play with these optimizer options for the second handin item, as described in the writeup
-- NOTE: see below for optimState, storing optimiser settings
local opt = {}         -- these options are used throughout
opt.optimization = 'sgd'
opt.batch_size = 64
opt.train_size = 8000  -- set to 0 or 60000 to use all 60000 training data
opt.test_size = 0      -- 0 means load all data
opt.epochs = 2         -- **approximate** number of passes through the training data (see below for the `iterations` variable, which is calculated from this)

-- NOTE: the code below changes the optimization algorithm used, and its settings
local optimState       -- stores a lua table with the optimization algorithm's settings, and state during iterations
local optimMethod      -- stores a function corresponding to the optimization routine
-- remember, the defaults below are not necessarily good
if opt.optimization == 'lbfgs' then
  optimState = {
    learningRate = 1e-1,
    maxIter = 2,
    nCorrection = 10
  }
  optimMethod = optim.lbfgs
elseif opt.optimization == 'sgd' then
  optimState = {
    learningRate = 1e-2,
    weightDecay = 0,
    momentum = 0.9,
    learningRateDecay = 1e-7
  }
  optimMethod = optim.sgd
elseif opt.optimization == 'adagrad' then
  optimState = {
    learningRate = 1e-1,
  }
  optimMethod = optim.adagrad
else
  error('Unknown optimizer')
end

mnist.download()       -- download dataset if not already there

-- load dataset using dataset-mnist.lua into tensors (first dim of data/labels ranges over data)
local function load_dataset(train_or_test, count)
    -- load
    local data
    if train_or_test == 'train' then
        data = mnist.loadTrainSet(count, {32, 32})
    else
        data = mnist.loadTestSet(count, {32, 32})
    end

    -- shuffle the dataset
    local shuffled_indices = torch.randperm(data.data:size(1)):long()
    -- creates a shuffled *copy*, with a new storage
    data.data = data.data:index(1, shuffled_indices):squeeze()
    data.labels = data.labels:index(1, shuffled_indices):squeeze()

    -- TODO: (optional) UNCOMMENT to display a training example
    -- for more, see torch gnuplot package documentation:
    -- https://github.com/torch/gnuplot#plotting-package-manual-with-gnuplot
    --gnuplot.imagesc(data.data[10])

    -- vectorize each 2D data point into 1D
    data.data = data.data:reshape(data.data:size(1), 32*32)

    print('--------------------------------')
    print(' loaded dataset "' .. train_or_test .. '"')
    print('inputs', data.data:size())
    print('targets', data.labels:size())
    print('--------------------------------')

    return data
end

local train = load_dataset('train', opt.train_size)
local test = load_dataset('test', opt.test_size)

------------------------------------------------------------------------------
-- MODEL
------------------------------------------------------------------------------

local n_train_data = train.data:size(1) -- number of training data
local n_inputs = train.data:size(2)     -- number of cols = number of dims of input (1024)
local n_outputs = train.labels:max()    -- highest label = # of classes

print(train.labels:max())
print(train.labels:min())

local lin_layer = nn.Linear(n_inputs, n_outputs) --从model func输入input得到output的linear layer
local softmax = nn.LogSoftMax() --softmax layer，得到两个class in sigmoid func in log form（没有e）
local model = nn.Sequential() --有多层的network
model:add(lin_layer) --加上linear layer：把input和parameters乘起来
model:add(softmax) --加上softmax layer：分出class

------------------------------------------------------------------------------
-- LOSS FUNCTION
------------------------------------------------------------------------------

local criterion = nn.ClassNLLCriterion() --negative log likelihood function: 这个function算的是这个NN的loss！！！

------------------------------------------------------------------------------
-- TRAINING
------------------------------------------------------------------------------

local parameters, gradParameters = model:getParameters()

------------------------------------------------------------------------
-- Define closure with mini-batches 
------------------------------------------------------------------------

local counter = 0
local feval = function(x) -- x here is parameters rather than input points
  if x ~= parameters then 
    parameters:copy(x)
  end

  -- get start/end indices for our minibatch (in this code we'll call a minibatch a "batch")
  --           ------- 
  --          |  ...  |
  --        ^ ---------<- start index = i * batchsize + 1
  --  batch | |       |
  --   size | | batch |       
  --        v |   i   |<- end index (inclusive) = start index + batchsize
  --          ---------                         = (i + 1) * batchsize + 1
  --          |  ...  |                 (except possibly for the last minibatch, we can't 
  --          --------                   let that one go past the end of the data, so we take a min())
  local start_index = counter * opt.batch_size + 1
  local end_index = math.min(n_train_data, (counter + 1) * opt.batch_size + 1) --防止minibatch就是整个dataset的最后一个point，再+1的话就out of range了。
  if end_index == n_train_data then --已经是最后一个data point
    counter = 0
  else
    counter = counter + 1 --not to the end, can continue to extract minibatch
  end

  local batch_inputs = train.data[{{start_index, end_index}, {}}]
  local batch_targets = train.labels[{{start_index, end_index}}]
  gradParameters:zero() --clear every time to avoid accumulating

  -- In order, these lines compute:
  -- 1. compute outputs (log probabilities) for each data point
  local batch_outputs = model:forward(batch_inputs)
  -- 2. compute the loss of these outputs, measured against the true labels in batch_target
  local batch_loss = criterion:forward(batch_outputs, batch_targets)
  -- 3. compute the derivative of the loss wrt the outputs of the model
  local dloss_doutput = criterion:backward(batch_outputs, batch_targets) 
  -- 4. use gradients to update weights, we'll understand this step more next week
  model:backward(batch_inputs, dloss_doutput)

  -- optim expects us to return
  --     loss, (gradient of loss with respect to the weights that we're optimizing)
  return batch_loss, gradParameters --gradient of loss wrt parameters
end
  
------------------------------------------------------------------------
-- OPTIMIZE: FIRST HANDIN ITEM
------------------------------------------------------------------------
local losses = {}    --table variable    -- training losses for each iteration/minibatch
local test_losses={} --test losses
local epochs = opt.epochs  -- number of full passes over all the training data--assume 10 rounds 【epochs是go through ALL THE DATA!!!】
local iterations = epochs * math.ceil(n_train_data / opt.batch_size--[[no. of minibatches]]) -- integer number of minibatches to process 【iteration是go through一个minibatch】
-- (note: number of training data might not be divisible by the batch size, so we round up)


-- In each iteration, we:
--    1. call the optimization routine, which
--      a. calls feval(parameters), which
--          i. grabs the next minibatch
--         ii. returns the loss value and the gradient of the loss wrt the parameters, evaluated on the minibatch
--      b. the optimization routine uses this gradient to adjust the parameters so as to reduce the loss.
--    3. then we append the loss to a table (list) and print it
for i = 1, iterations do
  -- optimMethod is a variable storing a function, either optim.sgd or optim.adagrad or ...
  -- see documentation for more information on what these functions do and return:
  --   https://github.com/torch/optim
  -- it returns (new_parameters, table), where table[0] is the value of the function being optimized【划重点】
  -- and we can ignore new_parameters because `parameters` is updated in-place every time we call 
  -- the optim module's function. It uses optimState to hide away its bookkeeping that it needs to do
  -- between iterations.
  local _, minibatch_loss = optim.sgd(feval, parameters, optimState)  --to use _ to ignore the output we don't care
  -- which is the parameter in this case. The only output we want is the loss
  -- Our loss function is cross-entropy, divided by the number of data points,
  -- therefore the units (units in the physics sense) of the loss is "loss per data sample".
  -- Since we evaluate the loss on a different minibatch each time, the loss will sometimes 
  -- fluctuate upwards slightly (i.e. the loss estimate is noisy).

  if i % 10 == 0 then -- don't print *every* iteration, this is enough to get the gist
      print(string.format("minibatches processed: %6s, loss = %6.6f", i, minibatch_loss[1]))
  --一个iteration go through一个minibatch，但是不用把每个minibatch的loss打出来，所以每十个minibatch （也就是每iterate10次）print out一次信息，内容是：1.现在是在run第几个minibatch；2.这个batch的training loss是多少。
  end
  -- TIP: use this same idea of not saving the test loss in every iteration if you want to increase speed.
  -- Then you can get, 10 (for example) times fewer values than the training loss. If you do this,
  -- you just have to be careful to give the correct x-values to the plotting function, rather than
  -- Tensor{1,2,...,#losses}. HINT: look up the torch.linspace function, and note that torch.range(1, #losses) 因为每10个batch才存了一次testing loss，所以画图时对应的x轴应该是每10个batch的最后一个data point而不是每一个data point都有对应的loss
  -- is the same as torch.linspace(1, #losses, #losses).

  losses[#losses + 1] = minibatch_loss[1] -- append the new training loss

 --[[local a=1;
 if i % 10 == 0 then
   
  a=a+1
--每优化一次模型就用test set测一次loss，然后存下了，可以用tip里的办法每十个存一次加快loop速度。]]
  -- 1. compute outputs (log probabilities) for test data point
  local test_out = model:forward(test.data)
  -- 2. compute the loss of these outputs, measured against the true labels in test_target
  local test_loss = criterion:forward(test_out, test.labels)
  -- 3. store the current testing loss to the total testing loss tensor
 --[[end ]]
  test_losses[#test_losses+1]=test_loss

  
end



-- TODO: for the first handin item, evaluate test loss above, and add to the plot below
--       see TIP/HINT above if you want to make the optimization loop faster


  

-- Turn table of losses into a torch Tensor, and plot it
gnuplot.plot({'train',
  torch.range(1, #losses),        -- x-coordinates for data to plot, creates a tensor holding {1,2,3,...,#losses}
  torch.Tensor(losses),           -- y-coordinates (the training losses)
  '-'},
{'test',
  torch.range(1, #test_losses),        -- x-coordinates for data to plot, creates a tensor holding {1,2,3,...,#losses}
  torch.Tensor(test_losses),           -- y-coordinates (the training losses)
  '-'})

------------------------------------------------------------------------------
-- TESTING THE LEARNED MODEL: 2ND HANDIN ITEM
------------------------------------------------------------------------------

local logProbs = model:forward(test.data)
local classProbabilities = torch.exp(logProbs)
local _, classPredictions = torch.max(classProbabilities, 2)
local classification_error = 0
-- TODO: compute test classification error here for the second handin item]]
--classification_error = criterion:forward(classPredictions, test.labels)
classification_error=torch.eq(classPrediction, test.lables:long() --[[把test lable从byte tensor转成long tensor，同类型的变量做比较]]):float():mean() --mean不一定是int所以cast到float
print(classification_error)
--print(classPredictions) --这里得到的数字就是predict分类出来的class，1,2,3....10
-- classPredictions holds predicted classes from 1-10

--或者是另一个办法，access到每一个label里面去比较
for i=1,(#classPredictions)[1] do
  if torch.eq(classPredictions[i], test.labels[i][1]) then
	classification_error = classification_error 
  else 
	classification_error = classificaton_error+1
  end
end
print(classification_error )



