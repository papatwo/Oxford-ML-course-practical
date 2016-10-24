--****************************MACHINE LEARNING 2015 PRACTICAL ASSIGNMENT 2***************************--
--***********************************TRINITY TERM 2015*******WEEK 3**********************************--

--****CODE - example-linear-regression.lua***********************************************************--
require 'torch'
require 'optim'
require 'nn'
										-- training data
data = torch.Tensor{{40,6,4},
		    {44,10,4},
		    {46,12,5},
    		    {48,14,7},
		    {52,16,9},
		    {58,18,12},
		    {60,22,14},
		    {68,24,20},
		    {74,26,21},
		    {80,32,24}} --{corn,fertilizer,insecticide}

model = nn.Sequential()                 -- define the container
ninputs = 2; noutputs = 1 				-- neuron, 2 inputs, 1 output
model:add(nn.Linear(ninputs, noutputs)) -- define the only module
criterion = nn.MSECriterion()			-- minimize mean squared error

x, dl_dx = model:getParameters() -- x = parameter vector; dl/dx = grads
feval = function(x_new) --eval loss function & grad
   if x ~= x_new then
      x:copy(x_new) -- check gradient descent formula. x is replaced by x_new if there's a 
   end	               -- smaller value than before
   
   _nidx_ = (_nidx_ or 0) + 1 		   -- select new training sample - a row of "data"
						-- 只是做_nidx_=_nidx+1. or 0不知道什么意思
   if _nidx_ > (#data)[1] then _nidx_ = 1 end
	--#data gives the dim of data, in this case 10x3
	--(#data)[1]=10, which is the no. rows of data
	--when _nidx_ exceeds 10, force it back to 1.

   local sample = data[_nidx_]		--eg. extract the 1st row of data: 40(corn)   6(ferilizer)   4(insecticide)
   local target = sample[{ {1} }]      -- extract 1st element in sample: 40 (要predict的东西)
   local inputs = sample[{ {2,3} }]    -- 2nd and 3rd element in sample: 6 and 4 (inputs)
   -- this funny looking syntax allows slicing of arrays.
	
   dl_dx:zero() -- reset gradients (otherwise always accumulated, to accomodate batch methods)

   -- evaluate the loss function and its derivative wrt x, for that sample
   local loss_x = criterion:forward(model:forward(inputs), target)
   model:backward(inputs, criterion:backward(model.output, target))
				-- forward算的是loss（吗）
				-- update gradient then accumulate parameters; backpropagation  
   return loss_x, dl_dx		-- return loss(x) and dloss/dx (updated by backpropagation)
end 

-- Given the function above, we can now easily train the model using SGD.
-- For that, we need to define four key parameters:
--   a learning rate: the size of the step taken at each stochastic 
--   estimate of the gradient
--   a weight decay, to regularize the solution (L2 regularization)
--   a momentum term, to average steps over time
--   a learning rate decay, to let the algorithm converge more precisely

sgd_params = {	--train model using SGD
   weightDecay = 0, learningRate = 1e-3,   --10^(-3); decay regularizes solution / L2
   momentum = 0, learningRateDecay = 1e-4, --10^(-4); momentum averages steps over time
}

-- We're now good to go... all we have left to do is run over the dataset
-- for a certain number of iterations, and perform a stochastic update 
-- at each iteration. The number of iterations is found empirically here,
-- but should typically be determinined using cross-validation.

--cycle over training data的次数，为了节约时间选1000次。1w次：1e4 is enough
for i = 1,1e3 do 			--number of epochs/full loops over training data
   current_loss = 0			--estimate average loss

   for i = 1,(#data)[1] do 		--loop over each row of training data
    

      -- optim contains several optimization algorithms. 
      -- All of these algorithms assume the same parameters:
      --   + a closure that computes the loss, and its gradient wrt to x, 
      --     given a point x
      --   + a point x
      --   + some parameters, which are algorithm-specific

      
      -- Functions in optim all return two things:
      --   + the new x, found by the optimization method (here SGD)
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.	
      params,fs = optim.sgd(feval,x,sgd_params) -- use optim's SGD algorithm, returns new x and the
						-- value of loss functions at all points used by
      current_loss = current_loss + fs[1]	-- algorithm. 即fs是所有点的loss值by using sgd这个算法
      --为什么要每次加上fs的第一个值？？？
   end

   current_loss = current_loss / (#data)[1]      -- report average error on epoch
   print('current loss = ' .. current_loss)
end

text = {40.32, 42.92, 45.33, 48.85, 52.37, 57, 61.82, 69.78, 72.19, 79.42}
print('id  approx   text')						-- testing trained model
for i = 1,(#data)[1] do
   local myPrediction = model:forward(data[i][{{2,3}}])
   print(string.format("%2d  %6.2f %6.2f", i, myPrediction[1], text[i]))
end

--************** HAND-IN ***************************************************************************--

-- Question 2: testing the trained model on new set:	   -- predictions:
dataTest = torch.Tensor{ {6,4},{10,5},{14,8}}	 	   -- #   1e4 epochs 1e3 epochs 1e5 epochs
print('parameters:')	-- print table head		   -- 1    40.10       33.37       40.33
print(params)						   -- 2    43.88       40.44       44.03
							   -- 3    49.89       47.06       49.96

print('id test')	-- predict and print test values   -- parameters: bias insecticide fertilizer
for i=1,(#dataTest)[1] do 				   -- 1e3 epochs: 23.21   1.8392     -0.24
   local prediction = model:forward(dataTest[i][{{1,2}}])  -- 1e4 epochs: 31.63   1.1146     0.6676
   print(string.format("%2d  %6.2f", i, prediction[1]))    -- 1e5 epochs: 31.98   1.11       0.64
end
--**************************************************************************************************--

--Question 3: least squares solution - no optimization here, just using formula & data provided
y = data:narrow(2,1,1) --slice first column, then rest of it, and construct biased data Tensor
X = torch.cat(torch.ones((#data)[1],1),data:narrow(2,2,2))   -- by adding bias column (ones)
XT= X:transpose(1,2)					     -- swap width (1) and height (2)

biasedDataTest = torch.cat(torch.ones((#dataTest)[1],1), dataTest)
print(biasedDataTest)			-- also add bias column to test data    PREDICTION OUTPUTS:
										--       1. 40.32
print('LSParams')				-- print out parameters 	--       2. 44.03
LSParams = (torch.inverse(XT * X))*XT*y						--       3. 49.96
print(LSParams)
					-- predict by tensor multiplication
LSPredictions = biasedDataTest * LSParams			       --bias insecticide fertilizer
print(LSPredictions)				-- resulting parameters: 31.98     1.11      0.65

-- note convergence as number of epochs increases to the prediction given by the least squares model

--[[--**************************************************************************************************--
--***OPTIONAL / ADVANCED: implementing ridge regression with different models***********************--
require 'torch'
require 'optim'
require 'nn'
require 'gnuplot'         -- plot outputs as well

								--***DATA GENERATION****************--
-- quadratic polynomial for data generation: f(x,y) = a + bx + cx^2; in one variable to make it easy
a = 2; b = -3; c = 1 	      -- set generation parameters here                               to plot.
nTrain = 1000;nTest  = 50     -- set number of training & testing instances
data = torch.Tensor(nTrain,1); labels = torch.Tensor(nTrain,1);         --training set
testData = torch.Tensor(nTest,1); testLabels = torch.Tensor(nTest,1);   --testing set

for i=1,nTrain do
   local x = torch.uniform(1,5)
   local noise = torch.normal(0,0.1) --add Gaussian noise; mean zero, variance 0.1
   data[i][1] = x
   labels[i][1]=a + b*x+c*x*x+noise
end
for i=1,nTest do
   local x = torch.uniform(1,5)
   local noise = torch.normal(0,0.1) --add Gaussian noise; mean zero, variance 0.1
   testData[i][1] = x
   testLabels[i][1]=a + b*x+c*x*x+noise
end
								--***SOLUTION TO PROBLEM STARTS HERE--
local function quadphi(x)        -- quadratic basis function;
   return torch.Tensor{1,x,x*x}
end
local function cubephi(x)        -- cubic basis function
   return torch.Tensor{1,x,x*x,x*x*x}
end
local function quartphi(x)       -- quartic basis function
   return torch.Tensor{1,x,x*x,x*x*x,x*x*x*x}
end

local quadPhi = torch.Tensor(nTrain,3)  --extract features using
local cubePhi = torch.Tensor(nTrain,4)  --the phi functions
local quartPhi = torch.Tensor(nTrain,5)
for i=1,nTrain do
   local phirow = quadphi(data[i][1])
   for j=1,3 do
      quadPhi[i][j]=phirow[j]
   end
   local phirow = cubephi(data[i][1])
   for j=1,4 do
      cubePhi[i][j]=phirow[j]
   end
   local phirow = quartphi(data[i][1])
   for j=1,5 do
      quartPhi[i][j]=phirow[j]
   end
end

local regularizerQuad = torch.mul(torch.eye(3),0.0001) --regularize
local regularizerCube = torch.mul(torch.eye(4),0.0001)
local regularizerQuar = torch.mul(torch.eye(5),0.0001)
								--transformed training data
local thetaQuad  = torch.inverse(((quadPhi:t())*quadPhi)+regularizerQuad)*(quadPhi:t())*labels
local thetaCube  = torch.inverse(((cubePhi:t())*cubePhi)+regularizerCube)*(cubePhi:t())*labels
local thetaQuart = torch.inverse(((quartPhi:t())*quartPhi)+regularizerQuar)*(quartPhi:t())*labels

X = torch.cat(torch.ones((#data)[1],1),data) --and add bias for linear model
biasedDataTest = torch.cat(torch.ones((#testData)[1],1), testData) --and to test data.
XT=X:transpose(1,2)    --transpose X by swapping width (1) & height (2)
LSParams = (torch.inverse(XT * X))*XT*labels --linear parameters

local quadPhiTest = torch.Tensor(nTest,3); local cubePhiTest = torch.Tensor(nTest,4)
local quartPhiTest = torch.Tensor(nTest,5)	--transformed testing data

for i=1,nTest do                                --apply basis functions to test data
   local phirow = quadphi(testData[i][1])    
   for j=1,3 do
      quadPhiTest[i][j]=phirow[j]
   end
   local phirow = cubephi(testData[i][1])
   for j=1,4 do
      cubePhiTest[i][j]=phirow[j]
   end
   local phirow = quartphi(testData[i][1])
   for j=1,5 do
      quartPhiTest[i][j]=phirow[j]
   end
end

local quadPred = quadPhiTest * thetaQuad   --quadratic model     (FIRST IMAGE)
local cubePred = cubePhiTest * thetaCube    --cubic model        (SECOND IMAGE)  
local quartPred= quartPhiTest* thetaQuart   --quartic model      (THIRD IMAGE)
local LSPred  = biasedDataTest * LSParams       --linear model   (FOURTH IMAGE)

print("Linear model predictions")      |     print("Linear parameters")
print(LSPred)                          |     print(LSParams)
print("Quadratic model predictions")   |     print("Quadratic parameters")
print(quadPred)                        |     print(thetaQuad)
print("Cubic model predictions")       |     print("Cubic parameters")
print(cubePred)                        |     print(thetaCube)
print("Quartic model predictions")     |     print("Quartic parameters")
print(quartPred)                       |     print(thetaQuart)
print("Actual values")                 |
print(testLabels)                      |
gnuplot.plot({torch.cat(testData,quadPred),'+'}, {torch.cat(testData,testLabels), '+'})
gnuplot.plot({torch.cat(testData,cubePred),'+'}, {torch.cat(testData,testLabels), '+'})
gnuplot.plot({torch.cat(testData,quartPred),'+'}, {torch.cat(testData,testLabels), '+'})
gnuplot.plot({torch.cat(testData,LSPred),'+'}, {torch.cat(testData,testLabels), '+'})
-- predictions in blue, actual values in green

--example output parameters: BIAS (1)    x     x^2   x^3   x^4		--note that higher
-- LINEAR: 			-5.78   3.04     			--degree polynomials
-- QUADRATIC: 			1.974  -2.99  0.99			--closely approximate
-- CUBIC:                       1.951  -2.96  0.99  0.01		--the generating
-- QUARTIC:			1.839  -2.77  0.88  0.03  -0.002	--quadratic on [1,5].]]
