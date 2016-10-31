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
--************** HAND-IN ***************************************************************************--
--Question 2
dataTest = torch.Tensor{
{6, 4},
{10, 5},
{14, 8}
}
print("epochs:1e4")
print("params:")
print(params)
print('id  approx   text')
for i=1,(#dataTest)[1] do
   local newPrediction = model:forward(dataTest[i][{{1,2}}])
   print(string.format("%2d	%6.2f", i, newPrediction[1]))
end



--************** HAND-IN ***************************************************************************--
--************** HAND-IN ***************************************************************************--
--Question 3
--Normal Equation: theta=(Xtranspose*X)inverse*Xtranspose*y
print("Question 3")
xin=data:narrow(2,2,2) --把input从data里extract出来，which is X
bias=torch.ones((#xin)[1]) --加上bias term
x=torch.cat(bias,xin,2) --what is the 2
y=data:narrow(2,1,1)
x_trans=x:transpose(2,1) --算X的transpose
xx_inv=torch.inverse(x_trans*x) --Xtrans*X的inverse
theta=xx_inv*x_trans*y

print('normal equation - method to solve parameters theta analytically:')
print(theta)
text = torch.Tensor{40.32, 42.92, 45.33, 48.85, 52.37, 57, 61.82, 69.78, 72.19, 79.42}
print('  approx   text')	
id=torch.range(1,10)					-- testing trained model
y_predict=x*theta
print(torch.cat(y_predict,text))
--print(string.format("%6.2f %6.2f",y_predict, text))	--打不出

--[[for i = 1,(#x)[1] do
   print(string.format("%2d  %6.2f %6.2f", i, y_predict[i], text[i]))
end]]




