require 'requ'

-- NOTE: Assumes input and output to module are 1-dimensional, i.e. doesn't test the module
--       in mini-batch mode. It's easy to modify it to do that if you want, though.
local function jacobian_wrt_input(module, x, eps)
  -- compute true Jacobian (rows = over outputs, cols = over inputs, as in our writeup's equations)
  local z = module:forward(x):clone() --x is input data
  local jac = torch.DoubleTensor(z:size(1), x:size(1)) --z is output data
  
  -- get true Jacobian, ROW BY ROW
  --selecting gradOutput to be a vector with only one 1 and the rest of the elements 0 lets you
  --select out one whole row, by giving this to backward or updateGradInput
  local one_hot = torch.zeros(z:size())
  for i = 1, z:size(1) do
    one_hot[i] = 1
    jac[i]:copy(module:backward(x, one_hot))--一行一行的扔进backward
    --backward实在算啥？(derivative wrt in?)根据p3的model:backward(input,dloss/dout)
    one_hot[i] = 0
  end
  
  -- compute finite-differences Jacobian, COLUMN BY COLUMN
  local jac_est = torch.DoubleTensor(z:size(1), x:size(1))
  for i = 1, x:size(1) do
    -- TODO: modify this to perform a two-sided estimate. Remember to do this carefully, because 
    --       nn modules reuse their output buffer across different calls to forward.

    --TWO-sided estimate 
    x[i] = x[i] + eps
    local z_plus = --[[module:forward(x) -- error will be very large]]torch.Tensor(z:size())
    z_plus:copy(module:forward(x)) -- in case output buffer was not cleared
    x[i] = x[i] - 2 * eps
    --local z_minus = torch.Tensor(z:size(1))
    jac_est[{{},i}]:copy(z_plus:add(-1, module:forward(x))):div( 2 * eps) 
------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------
   --[[ -- ONE-sided estimate
    x[i] = x[i] + eps
    local z_offset = module:forward(x)
    x[i] = x[i] - eps
    jac_est[{{},i}]:copy(z_offset):add(-1, z):div(eps) ]]--a:add(value,b):Multiply elements of tensor b by the scalar value and add it to tensor a. The number of elements must match, but sizes do not matter.
  end

  -- computes (symmetric) relative error of gradient
  local abs_diff = (jac - jac_est):abs()
  return jac, jac_est, torch.mean(abs_diff), torch.min(abs_diff), torch.max(abs_diff)
end

---------------------------------------------------------
-- test our layer in isolation
--
torch.manualSeed(1)
local requ = nn.ReQU()

local x = torch.randn(10) -- random input to layer
print(x)
print(jacobian_wrt_input(requ, x, 1e-6))

-- two-sided error 6e-12
-- one-sided error 6e-8