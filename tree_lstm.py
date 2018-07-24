import torch
import torch.nn as nn
import torch.nn.functional as F

#from . import Constants


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, cuda):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.words = []
        self.states = []
        self.cuda = cuda
        '''Attention'''
        self.attention = True
        self.Wh = nn.Linear(mem_dim,mem_dim)
        self.Us = nn.Linear(mem_dim,mem_dim)
        self.map_alpha = nn.Linear(mem_dim, mem_dim)

        if cuda:
            self.w = nn.Parameter(torch.randn(1,mem_dim).cuda(), requires_grad = True)
        else:
            self.w = nn.Parameter(torch.randn(1,mem_dim), requires_grad = True)

    def node_forward(self, inputs, child_c, child_h):
        if self.attention == True:
            if self.cuda:
                 m = torch.zeros(child_h.size()).cuda()
            else:
                 m = torch.zeros(child_h.size())
            expo = []
    
            for i in range(int(m.size(0))):
                m[i] = F.tanh(self.Wh(child_h[i]) + self.Us(inputs.unsqueeze(0))).squeeze()
                expo.append(torch.mm(self.w, m[i].unsqueeze(1)))
            
            if self.cuda:
                expo = torch.tensor(expo).unsqueeze(0).cuda()
            else:
                expo = torch.tensor(expo).unsqueeze(0)
            alpha = F.softmax(expo, dim=-1)
            child_h_sum = F.tanh(self.map_alpha(torch.mm(alpha, child_h)))
        else:
            child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs, sent):
        self.words.append(tree.idx)
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs, sent)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        self.states.append(tree.state[1])
        return tree.state


# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)

        out = F.sigmoid(self.wh(vec_dist))
        out = F.log_softmax(self.wp(out), dim=1)
        return out

class BinaryTreeLeafModule(nn.Module):
    """
  local input = nn.Identity()()
  local c = nn.Linear(self.in_dim, self.mem_dim)(input)
  local h
  if self.gate_output then
    local o = nn.Sigmoid()(nn.Linear(self.in_dim, self.mem_dim)(input))
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end

  local leaf_module = nn.gModule({input}, {c, h})
    """
    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeLeafModule, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.cx = nn.Linear(self.in_dim, self.mem_dim) # 200 300
        self.ox = nn.Linear(self.in_dim, self.mem_dim) # 200 300
        if self.cudaFlag:
            self.cx = self.cx.cuda()
            self.ox = self.ox.cuda()

    def forward(self, input):
#        print(input.size()) # [200]
        c = self.cx(input) # 300
        o = F.sigmoid(self.ox(input)) # 300 (0~1)
        h = o * F.tanh(c) #300
        return c, h # 300 300

class BinaryTreeComposer(nn.Module):
    """
  local lc, lh = nn.Identity()(), nn.Identity()()
  local rc, rh = nn.Identity()(), nn.Identity()()
  local new_gate = function()
    return nn.CAddTable(){
      nn.Linear(self.mem_dim, self.mem_dim)(lh),
      nn.Linear(self.mem_dim, self.mem_dim)(rh)
    }
  end

  local i = nn.Sigmoid()(new_gate())    -- input gate
  local lf = nn.Sigmoid()(new_gate())   -- left forget gate
  local rf = nn.Sigmoid()(new_gate())   -- right forget gate
  local update = nn.Tanh()(new_gate())  -- memory cell update vector
  local c = nn.CAddTable(){             -- memory cell
      nn.CMulTable(){i, update},
      nn.CMulTable(){lf, lc},
      nn.CMulTable(){rf, rc}
    }

  local h
  if self.gate_output then
    local o = nn.Sigmoid()(new_gate()) -- output gate
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end
  local composer = nn.gModule(
    {lc, lh, rc, rh},
    {c, h})    
    """
    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeComposer, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        if cuda:
            self.w = nn.Parameter(torch.randn(1,mem_dim).cuda(), requires_grad = True)
        else:
            self.w = nn.Parameter(torch.randn(1,mem_dim), requires_grad = True)
        
        def new_gate():
            lh = nn.Linear(self.mem_dim, self.mem_dim)
            rh = nn.Linear(self.mem_dim, self.mem_dim)
            return lh, rh

        self.ilh, self.irh = new_gate()
        self.lflh, self.lfrh = new_gate()
        self.rflh, self.rfrh = new_gate()
        self.ulh, self.urh = new_gate()
        
        '''Attention'''
        self.Wh = nn.Linear(mem_dim,mem_dim)
        self.Us = nn.Linear(mem_dim,mem_dim)
        self.map_alpha = nn.Linear(mem_dim, mem_dim)
        
        if self.cudaFlag:
            self.ilh = self.ilh.cuda()
            self.irh = self.irh.cuda()
            self.lflh = self.lflh.cuda()
            self.lfrh = self.lfrh.cuda()
            self.rflh = self.rflh.cuda()
            self.rfrh = self.rfrh.cuda()
            self.ulh = self.ulh.cuda()

    def forward(self, lc, lh , rc, rh, S):
#        print(lc.size(), lh.size(), rc.size(), rh.size(), S.size()) #  300 300 300 300 300
    
        ''' weight for child 1'''
        m_left = F.tanh(self.Wh(lh) + self.Us(S)).squeeze()

        ''' weight for child 2'''
        m_right = F.tanh(self.Wh(rh) + self.Us(S)).squeeze()

#        print(self.w.size(), m_left.size(), m_right.size())
        exp_left = torch.mm(self.w, m_left.unsqueeze(1)) # 1 300 300 1
        exp_right = torch.mm(self.w, m_right.unsqueeze(1)) # 1 300 300 1


        left_alpha = exp_left/(exp_left + exp_right)
        right_alpha = exp_right/(exp_left + exp_right)
        
        left_alpha = F.tanh(self.map_alpha(left_alpha * lh))
        right_alpha = F.tanh(self.map_alpha(right_alpha * rh))
        
        i = F.sigmoid(self.ilh(left_alpha) + self.irh(right_alpha))
        lf = F.sigmoid(self.lflh(left_alpha) + self.lfrh(right_alpha))
        rf = F.sigmoid(self.rflh(left_alpha) + self.rfrh(right_alpha))
        update = F.tanh(self.ulh(left_alpha) + self.urh(right_alpha))
        c =  i * update + lf*lc + rf*rc
        h = F.tanh(c)
#        print("ok")
        return c, h


class BinaryTreeComposer_noattn(nn.Module):
    """
  local lc, lh = nn.Identity()(), nn.Identity()()
  local rc, rh = nn.Identity()(), nn.Identity()()
  local new_gate = function()
    return nn.CAddTable(){
      nn.Linear(self.mem_dim, self.mem_dim)(lh),
      nn.Linear(self.mem_dim, self.mem_dim)(rh)
    }
  end

  local i = nn.Sigmoid()(new_gate())    -- input gate
  local lf = nn.Sigmoid()(new_gate())   -- left forget gate
  local rf = nn.Sigmoid()(new_gate())   -- right forget gate
  local update = nn.Tanh()(new_gate())  -- memory cell update vector
  local c = nn.CAddTable(){             -- memory cell
      nn.CMulTable(){i, update},
      nn.CMulTable(){lf, lc},
      nn.CMulTable(){rf, rc}
    }

  local h
  if self.gate_output then
    local o = nn.Sigmoid()(new_gate()) -- output gate
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end
  local composer = nn.gModule(
    {lc, lh, rc, rh},
    {c, h})    
    """
    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeComposer_noattn, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        def new_gate():
            lh = nn.Linear(self.mem_dim, self.mem_dim)
            rh = nn.Linear(self.mem_dim, self.mem_dim)
            return lh, rh

        self.ilh, self.irh = new_gate()
        self.lflh, self.lfrh = new_gate()
        self.rflh, self.rfrh = new_gate()
        self.ulh, self.urh = new_gate()

        if self.cudaFlag:
            self.ilh = self.ilh.cuda()
            self.irh = self.irh.cuda()
            self.lflh = self.lflh.cuda()
            self.lfrh = self.lfrh.cuda()
            self.rflh = self.rflh.cuda()
            self.rfrh = self.rfrh.cuda()
            self.ulh = self.ulh.cuda()

    def forward(self, lc, lh , rc, rh, S):
        i = F.sigmoid(self.ilh(lh) + self.irh(rh))
        lf = F.sigmoid(self.lflh(lh) + self.lfrh(rh))
        rf = F.sigmoid(self.rflh(lh) + self.rfrh(rh))
        update = F.tanh(self.ulh(lh) + self.urh(rh))
        c =  i* update + lf*lc + rf*rc
        h = F.tanh(c)
        return c, h




class BinaryTreeLSTM(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim): # 200 300
        super(BinaryTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.criterion = nn.BCELoss()

        self.leaf_module = BinaryTreeLeafModule(cuda,in_dim, mem_dim)
        self.composer = BinaryTreeComposer_noattn(cuda, in_dim, mem_dim)
        self.output_module = None

    def set_output_module(self, output_module):
        self.output_module = output_module

    def getParameters(self):
        """
        Get flatParameters
        note that getParameters and parameters is not equal in this case
        getParameters do not get parameters of output module
        :return: 1d tensor
        """
        params = []
        for m in [self.ix, self.ih, self.fx, self.fh, self.ox, self.oh, self.ux, self.uh]:
            # we do not get param of output module
            l = list(m.parameters())
            params.extend(l)

        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        return params

    def forward(self, tree, embs, sent, training = False):
        # add singleton dimension for future call to node_forward
        # embs = F.torch.unsqueeze(self.emb(inputs),1)
#        print("in forward")

        loss = torch.zeros(1) # init zero loss
        if self.cudaFlag:
            loss = loss.cuda()

        if tree.num_children == 0:
            # leaf case
#            print("lea encountered")
            tree.state = self.leaf_module.forward(embs[tree.idx-1]) # 300 300
        else:
            for idx in range(tree.num_children):
#                print("child: ", idx)
                _, child_loss = self.forward(tree.children[idx], embs, sent, training)
                loss = loss + child_loss
#            aasas
            lc, lh, rc, rh = self.get_child_state(tree)
            tree.state = self.composer.forward(lc, lh, rc, rh, sent)

        if self.output_module != None:
            output = self.output_module.forward(tree.state[1], training)
            tree.output = output
            if training and tree.gold_label != None:
                target = Var(utils.map_label_to_target_sentiment(tree.gold_label))
                if self.cudaFlag:
                    target = target.cuda()
#                loss = loss + self.criterion(output, target)
        return tree.state


    def get_child_state(self, tree):
        lc, lh = tree.children[0].state
        rc, rh = tree.children[1].state
        return lc, lh, rc, rh
