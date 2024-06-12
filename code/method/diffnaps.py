
import sys 
sys.path.append("./")
sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import method.dataLoader as mydl
import method.my_layers as myla
import method.my_loss as mylo




## not used
def initWeights(w, data):
    init.constant_(w, 0.01) 
    

class DiffnapsNet(nn.Module):
    def __init__(self, init_weights, label_dim, init_bias, data_sparsity, device_cpu, device_gpu, config=None):
        super(DiffnapsNet, self).__init__()
        input_dim = init_weights.size()[1]
        hidden_dim = init_weights.size()[0]
        # Initialization of the architecture fc0_enc is W^e and fc3_dec is W^d
        self.fc0_enc = myla.BinarizedLinearModule(input_dim, hidden_dim, .5, data_sparsity, False, init_weights, None, init_bias, device_cpu, device_gpu)
        self.fc3_dec = myla.BinarizedLinearModule(hidden_dim, input_dim, .5, data_sparsity, True, self.fc0_enc.weight.data, self.fc0_enc.weightB.data, None, device_cpu, device_gpu)
        self.act0 = myla.BinaryActivation(hidden_dim, device_gpu)
        self.act3 = myla.BinaryActivation(input_dim, device_gpu)
        torch.nn.init.xavier_normal_(self.fc0_enc.weight)

        self.classifier = nn.Linear(hidden_dim, label_dim,bias=False) #  corresponds to W^c

        if config.init_enc=="bimodal":
            print("BiModal")
            init_bi_modal(self.classifier.weight,0.25,0.75,0.1, device_cpu)
        else: 
            torch.nn.init.xavier_normal_(self.classifier.weight)
        self.bin_classifier = nn.Linear(hidden_dim, label_dim,bias=False)
        print(self.fc0_enc.weight.mean())



    def forward(self, x):
        x = self.fc0_enc(x)
        z = self.act0(x, False)
        classification = self.classifier(z)
        x = self.fc3_dec(z)
        output = self.act3(x, True)
        return output, classification, z

    def clipWeights(self, mini=-1, maxi=1):
        self.fc0_enc.clipWeights(mini, maxi)
        #self.classifier.clipWeights(mini, maxi)
        self.classifier.weight.data = self.classifier.weight.data.clamp(0,1)
        self.act0.clipBias()
        self.act3.noBias()
    
    def forward_test(self, x, t_enc, t_class): # Forwarding with binarized network
        w_bin = myla.BinarizeTensorThresh(self.classifier.weight,t_class)
        self.bin_classifier.weight.data = w_bin
        w_bin = myla.BinarizeTensorThresh(self.fc0_enc.weight,t_enc)
        x = self.fc0_enc.forward_test(x, t_enc)
        z = self.act0(x, False)
        classification = self.bin_classifier(z)
        x = self.fc3_dec(z)
        output = self.act3(x, True)
        return output, classification, z

    def forward_c_bin(self, x, t_class):
        w_bin = myla.BinarizeTensorThresh(self.classifier.weight,t_class)
        self.bin_classifier.weight.data = w_bin
        x = self.fc0_enc.forward(x)
        z = self.act0(x, False)
        classification = self.bin_classifier(z)
        x = self.fc3_dec(z)
        output = self.act3(x, True)
        return output, classification, z


    def learn(self, device_cpu, device_gpu, train_loader, optimizer, lossFun, epoch, log_interval, config, verbose=True):
        self.clipWeights()
        self.train()
        classification_loss = nn.CrossEntropyLoss()
        lossFun.config = config
        print_gpu(1)
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device_gpu)
            target = target.to(device_gpu)
            output, classification, z = self(data)
            itEW = [par for name, par in self.named_parameters() if name.endswith("enc.weight")]
            recon_loss = lossFun(output, data, next(iter(itEW)),hidden=z)
            #print(recon_loss)
            c_loss = classification_loss(classification,target)
            c_w = self.classifier.weight
            #print(config.lambda_c * c_loss )
            loss = recon_loss + config.lambda_c * c_loss + mylo.elb_regu_class(config.class_elb_k, config.class_elb_lamb, c_w, None) + mylo.horizontal_L2_class(config.wd_class, c_w, None)
            #print(mylo.elb_regu(config.class_elb_k, config.class_elb_lamb, c_w, None))
            loss.backward()
            optimizer.step()
            self.clipWeights()
            if batch_idx % log_interval == 0 and verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            optimizer.zero_grad()
        #print("End")
        return

def init_bi_modal(weight,m1,m2,std, device):
        left = torch.normal(mean=m1,std=std, size=weight.data.shape)
        right = torch.normal(mean=m2,std=std, size=weight.data.shape)
        mask = torch.randint(0,2,size=weight.data.shape)
        weight.data = (left*mask + right*(1-mask)).to(device)


def test(model, device_cpu, device_gpu, test_loader, lossFun, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    correct_class = 0
    numel = 0
    rows = 0
    classification_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device_gpu)
            target = target.to(device_gpu)
            output, classification, z = model(data)
            itEW = [par for name, par in model.named_parameters() if name.endswith("enc.weight")]
            test_loss += lossFun(output, data, next(iter(itEW))) + classification_loss(classification,target) 
            numel +=  output.numel()
            correct += torch.sum(output==data)
            correct_class += torch.sum(torch.argmax(classification.softmax(dim=1),dim=1)==target)
            rows += target.numel()

    _, target = next(iter(test_loader))
    if verbose:
        print('\nTest set: Average loss: {:.6f}, Recon Accuracy: {}/{} ({:.0f}%), Classification Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, numel, 100. * correct / numel, correct_class, rows, 100. * correct_class / rows))



def test_bin(model, device_cpu, device_gpu, test_loader, t_enc=0.3, t_class=0.9):
    model.eval()
    test_loss = 0
    correct = 0
    correct_class = 0
    numel = 0
    rows = 0
    classification_loss = nn.CrossEntropyLoss()
    incorret = []
    gt = []
    incorret_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device_gpu)
            target = target.to(device_gpu)
            output, classification, z = model.forward_test(data,t_enc, t_class)
            itEW = [par for name, par in model.named_parameters() if name.endswith("enc.weight")]
            #test_loss += lossFun(output, data, next(iter(itEW))) + classification_loss(classification,target) 
            numel +=  output.numel()
            correct += torch.sum(output==data)
            correct_class += torch.sum(torch.argmax(classification.softmax(dim=1),dim=1)==target)
            rows += target.numel()
            classi = torch.argmax(classification.softmax(dim=1),dim=1)
            #print(classi.shape)
            #print(target[ind].shape)
            ind = torch.argmax(classification.softmax(dim=1),dim=1)!=target
            incorret.append(data[ind].cpu().numpy())
            gt.append(target[ind].cpu().numpy())
            incorret_pred.append(classi[ind].cpu().numpy())
    _, target = next(iter(test_loader))
    print('\nTest set: Average loss: {:.6f}, Recon Accuracy: {}/{} ({:.0f}%), Classification Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, numel, 100. * correct / numel, correct_class, rows, 100. * correct_class / rows))
    return incorret, incorret_pred, gt

def test_c_bin(model, device_cpu, device_gpu, test_loader, t_enc=0.3, t_class=0.9):
    model.eval()
    test_loss = 0
    correct = 0
    correct_class = 0
    numel = 0
    rows = 0
    classification_loss = nn.CrossEntropyLoss()
    incorret = []
    gt = []
    incorret_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device_gpu)
            target = target.to(device_gpu)
            output, classification, z = model.forward_c_bin(data,t_class)
            itEW = [par for name, par in model.named_parameters() if name.endswith("enc.weight")]
            #test_loss += lossFun(output, data, next(iter(itEW))) + classification_loss(classification,target) 
            numel +=  output.numel()
            correct += torch.sum(output==data)
            correct_class += torch.sum(torch.argmax(classification.softmax(dim=1),dim=1)==target)
            rows += target.numel()
            classi = torch.argmax(classification.softmax(dim=1),dim=1)
            #print(classi.shape)
            #print(target[ind].shape)
            ind = torch.argmax(classification.softmax(dim=1),dim=1)!=target
            incorret.append(data[ind].cpu().numpy())
            gt.append(target[ind].cpu().numpy())
            incorret_pred.append(classi[ind].cpu().numpy())
    _, target = next(iter(test_loader))
    print('\nTest set: Average loss: {:.6f}, Recon Accuracy: {}/{} ({:.0f}%), Classification Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, numel, 100. * correct / numel, correct_class, rows, 100. * correct_class / rows))
    return incorret, incorret_pred, gt

def test_normal(model, device_cpu, device_gpu, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    correct_class = 0
    numel = 0
    rows = 0
    classification_loss = nn.CrossEntropyLoss()
    incorret = []
    gt = []
    incorret_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device_gpu)
            target = target.to(device_gpu)
            output, classification, z = model.forward(data)
            itEW = [par for name, par in model.named_parameters() if name.endswith("enc.weight")]
            #test_loss += lossFun(output, data, next(iter(itEW))) + classification_loss(classification,target) 
            numel +=  output.numel()
            correct += torch.sum(output==data)
            correct_class += torch.sum(torch.argmax(classification.softmax(dim=1),dim=1)==target)
            rows += target.numel()
            classi = torch.argmax(classification.softmax(dim=1),dim=1)
            #print(classi.shape)
            #print(target[ind].shape)
            ind = torch.argmax(classification.softmax(dim=1),dim=1)!=target
            incorret.append(data[ind].cpu().numpy())
            gt.append(target[ind].cpu().numpy())
            incorret_pred.append(classi[ind].cpu().numpy())
    _, target = next(iter(test_loader))
    print('\nTest set: Average loss: {:.6f}, Recon Accuracy: {}/{} ({:.0f}%), Classification Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, numel, 100. * correct / numel, correct_class, rows, 100. * correct_class / rows))
    return incorret, incorret_pred, gt

def update_elb(config):
    config.elb_lamb =  config.elb_lamb * config.regu_rate
    config.elb_k =  config.elb_k * config.regu_rate
    config.class_elb_lamb =  config.class_elb_lamb * config.class_regu_rate
    config.class_elb_k =  config.class_elb_k * config.class_regu_rate

def print_gpu(debug=0):
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    f = (t-r)/1024/1024
    #print("Position %d"%debug)
    #print(round(f,2))

def learn_diffnaps_net(data, config, labels = None, ret_test=False, verbose=True):
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.thread_num)
    device_cpu = torch.device("cpu")

    if not torch.cuda.is_available():
        device_gpu = device_cpu
        print("WARNING: Running purely on CPU. Slow.")
    else:
        device_gpu = torch.device("cuda")
    if labels is None:
        data_copy = np.copy(data)[:,:-2]
        labels_copy = (data[:,-2] + 2*data[:,-1]).astype(int)
    else:
        data_copy = data
        labels_copy = labels
    
    trainDS = mydl.DiffnapsDatDataset("file", config.train_set_size, True, device_cpu, data=data_copy, labels = labels_copy)
    train_loader = torch.utils.data.DataLoader(trainDS, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mydl.DiffnapsDatDataset("file", config.train_set_size, False, device_cpu, data=data_copy, labels = labels_copy), batch_size=config.test_batch_size, shuffle=True)
    
    if config.hidden_dim == -1:
        hidden_dim = trainDS.ncol()
        
    new_weights = torch.zeros(config.hidden_dim, trainDS.ncol(), device=device_gpu)
    initWeights(new_weights, trainDS.data)
    new_weights.clamp_(1/(trainDS.ncol()), 1)
    bInit = torch.zeros(config.hidden_dim, device=device_gpu)
    init.constant_(bInit, -1)
    
    model = config.model(new_weights, np.max(labels_copy)+1, bInit, trainDS.getSparsity(), device_cpu, device_gpu, config=config).to(device_gpu)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    lossFun = mylo.weightedXor(trainDS.getSparsity(), config.weight_decay, device_gpu, label_decay = 0, labels=2)

    scheduler = MultiStepLR(optimizer, [5,7], gamma=config.gamma)

    print_gpu()
    for epoch in range(1, config.epochs + 1):
        #print(model.fc0_enc.weight.data.mean())
        model.learn(device_cpu, device_gpu, train_loader, optimizer, lossFun, epoch, config.log_interval, config, verbose=verbose)
        test(model, device_cpu, device_gpu, test_loader, lossFun,verbose=verbose)
        scheduler.step()
        update_elb(config)
    if ret_test:
        return model, model.fc0_enc.weight.data, trainDS, test_loader
    else:
        return model, model.fc0_enc.weight.data, trainDS