import torch
import torch.nn.functional as F


# 定义 Grad-CAM 类
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        for param in model.parameters():
            param.requires_grad = False
        self.model.eval()
    def _register_hooks(self, grad):
        # 记录梯度
        self.gradients = grad
        # return grad_input  # 返回原始梯度        

    def generate_cam(self, input_tensor, cell_graph, gene_graph, target_class=None):
        
        # 前向传播
        input_tensor.requires_grad = True
        input_tensor.retain_grad()
        input_tensor.register_hook(self._register_hooks)
        self.model.zero_grad()
        output = self.model(input_tensor, cell_graph, gene_graph)['pred']
        # output = self.model(input_tensor, cell_graph, gene_graph)
       # 计算目标类别的梯度，使用 gradient 参数
        # 需要创建一个与输出大小相同的梯度张量
        one_hot_output = torch.zeros_like(output).to(self.device)
        one_hot_output[:, target_class] = 1  # 只关注目标类别
        output.backward(gradient=one_hot_output, retain_graph=True)

        
        # 获取梯度和激活值
        activations = input_tensor.detach().cpu()
        gradients = self.gradients.detach().cpu()
        cam = F.relu(gradients*activations).numpy()
        cell_cam = F.relu((gradients*activations).sum(axis=1, keepdims=True)).numpy()
        input_tensor.grad.zero_()
        # cam = cam/np.max(cam, axis=1,keepdims=True)
        # cam[np.isnan(cam)] = 0 
        return cam,cell_cam