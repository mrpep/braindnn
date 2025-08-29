#layer_map contains for each model the keys of the activation files that will be used.
############################################################################################################

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
                        'ReLU()--8'],
            'DS2': ['Hardtanh(min_val=0, max_val=20)--0',
                    'Hardtanh(min_val=0, max_val=20)--1',
                    'LSTM(1312, 1024, bidirectional=True)--0--cell',
                    'LSTM(1024, 1024, bidirectional=True)--0--cell',
                    'LSTM(1024, 1024, bidirectional=True)--1--cell',
                    'LSTM(1024, 1024, bidirectional=True)--2--cell',
                    'LSTM(1024, 1024, bidirectional=True)--3--cell',
                    'Linear(in_features=1024, out_features=29, bias=False)--0'],
                    }
kell_layers = ['relu0', 'maxpool0', 'relu1', 'maxpool1', 'relu2', 'relu3', 'relu4', 'avgpool', 'relufc']
resnet_layers = ['conv1_relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
kell_suffixes = ['audioset', 'multitask', 'speaker', 'word', 'music']
kell_map = {f'Kell2018{k}': kell_layers for k in kell_suffixes}
resnet_map = {f'ResNet50{k}': resnet_layers for k in kell_suffixes}
music_layers = ['relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
resnet_map['ResNet50music'] = music_layers
layer_map.update(kell_map)
layer_map.update(resnet_map)
layer_map['mel256-ec-large'] = [0,2,4,6,8,10,12,14,16,18,19]
layer_map['mel256-ec-large_st-nopn'] = [0,2,4,6,8,10,12,14,16,18,19]
layer_map['dasheng_06B'] = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,31]
layer_map['dasheng_12B'] = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,39]

#############################################################################################################
