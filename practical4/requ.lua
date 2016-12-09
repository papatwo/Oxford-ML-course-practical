require 'nn'

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)
  -- TODO
  self.output:resizeAs(input):copy(input)
  self.output:cmul(torch.gt(input, 0):double()):cmul(input)
-- (>) torch.gt第一个argument和第二个比大小, return 1（一>二）or 0 （一>二）  
-- 根据ReQU的operation把formula写下来。cmul是element-wise multiplication
-- 因为第一行已经copy了一次input放在self.output里，所以cmul一次input就好了
  return self.output
end

function ReQU:updateGradInput(input, gradOutput) --gradOutout from criterion:backward???
  -- TODO
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  --self.gradInput:cmul(torch.gt(gradOutput, 0):double()):mul(2):cmul(input) --torch.gt还是要比较input和0的大小而不是gradOutput
  self.gradInput:cmul(torch.gt(input, 0):double()):mul(2):cmul(input)
-- z=x^2 or z=0;
-- 当z=0, dloss/dz(gradOutput)=0 再次通过gt来判断是0还是x^2
-- 当z=x^2, dloss/dz=gradOutput; dz/dx=2x; 根据chain rule
-- dloss/dz*dz/dx=gradInput
  return self.gradInput
end

