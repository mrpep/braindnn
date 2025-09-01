#layer_map contains for each model the keys of the activation files that will be used.
############################################################################################################
ALL_MODELS = ['AST',
              'BEATs_iter1','BEATs_iter2','BEATs_iter3_finetuned_on_AS2M_cpt1','BEATs_iter3',
              'dasheng_06B', 'dasheng_12B', 'dasheng_base_ft-as','dasheng_base',
              'DCASE2020', 'DS2',
              'ec-ec-base', 
              'Kell2018audioset', 'Kell2018multitask', 'Kell2018music', 'Kell2018speaker', 'Kell2018word',
              'mel256-ec-base_st-nopn',
              'mel256-ec-base-as',
              'mel256-ec-base-fma',
              'mel256-ec-base-ll',
              'mel256-ec-base',
              'mel256-ec-large_st-nopn',
              'mel256-ec-large',
              'mel256-ec-small',
              'ResNet50audioset', 'ResNet50multitask','ResNet50music','ResNet50speaker','ResNet50word',
              'sepformer',
              'spec-ec-base',
              'spectemp',
              'VGGish',
              'wav2vec',
              'wav2vec2',
              'ZeroSpeech2020']
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

m_to_label = {'ResNet50multitask': 'ResNet50-Multitask',
              'ResNet50audioset': 'ResNet50-Audioset',
              'ResNet50word': 'ResNet50-Word',
              'ResNet50speaker': 'ResNet50-Speaker',
              'ResNet50music': 'ResNet50-Music',
              'Kell2018multitask': 'Kell2018-Multitask',
              'Kell2018speaker': 'Kell2018-Speaker',
              'Kell2018audioset': 'Kell2018-Audioset',
              'Kell2018word': 'Kell2018-Word',
              'Kell2018music': 'Kell2018-Music',           
              'BEATs_iter3': 'BEATs (Iter 3)',
              'BEATs_iter3_finetuned_on_AS2M_cpt1': 'BEATs FT',
              'BEATs_iter2': 'BEATs (Iter 2)',
              'BEATs_iter1': 'BEATs (Iter 1)',
              'spec-ec-base': 'Spec→EC',
              'ec-ec-base': 'EC→EC',
              'mel256-ec-base': 'Mel256→EC',
              'mel256-ec-small': 'Mel256→EC (Small)',
              'mel256-ec-base_st-nopn': 'Mel256→EC (ST)',
              'mel256-ec-base-fma': 'Mel256→EC (FMA)',
              'mel256-ec-base-ll': 'Mel256→EC (LL)',
              'mel256-ec-base-as': 'Mel256→EC (AS)',
              'mel256-ec-large': 'Mel256→EC (Large)',
              'mel256-ec-large_st-nopn': 'Mel256→EC (Large + ST)',
              'dasheng_base': 'Dasheng (Base)',
              'dasheng_06B': 'Dasheng (0.6B)',
              'dasheng_12B': 'Dasheng (1.2B)',
              'dasheng_base_ft-as': 'Dasheng FT',
              'wav2vec': 'Wav2Vec 2.0',
              'sepformer': 'Sepformer',
              'metricGAN': 'MetricGAN'}