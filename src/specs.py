layer_map = {'spectemp': ['avgpool'],
             'VGGish': ['ReLU()--0',
                        'MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)--0',
                        'ReLU()--1',
                        'MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)--1',
                        'ReLU()--2',
                        'ReLU()--3',
                        'MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)--2',
                        'ReLU()--4',
                        'ReLU()--5',
                        'MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)--3',
                        'ReLU()--6',
                        'ReLU()--7',
                        'ReLU()--8']}
kell_layers = ['relu0', 'maxpool0', 'relu1', 'maxpool1', 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc']
resnet_layers = ['conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
kell_suffixes = ['audioset', 'multitask', 'speaker', 'word', 'music']
kell_map = {f'Kell2018{k}': kell_layers for k in kell_suffixes}
resnet_map = {f'ResNet50{k}': resnet_layers for k in kell_suffixes}
layer_map.update(kell_map)
layer_map.update(resnet_map)
