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
kell_suffixes = ['audioset', 'word_speaker_audioset', 'speaker', 'word', 'music']
kell_map = {f'braindnn_kell2018_{k}': [f'braindnn_kell2018_{k}_{l}' for l in kell_layers] for k in kell_suffixes}
resnet_map = {f'braindnn_resnet50_{k}': [f'braindnn_resnet50_{k}_{l}' for l in resnet_layers] for k in kell_suffixes}
music_layers = ['relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
resnet_map['braindnn_resnet50_music'] = [f'braindnn_resnet50_music_{l}' for l in music_layers]
layer_map.update(kell_map)
layer_map.update(resnet_map)

layer_map['mel256-ec-base'] = ['mel256-ec-base_{}'.format(i) for i in range(10)]
layer_map['mel256-ec-base-as'] = ['mel256-ec-base-as_{}'.format(i) for i in range(10)]
layer_map['mel256-ec-base-fma'] = ['mel256-ec-base-fma_{}'.format(i) for i in range(10)]
layer_map['mel256-ec-base-ll'] = ['mel256-ec-base-ll_{}'.format(i) for i in range(10)]
layer_map['mel256-ec-base_st-nopn'] = ['mel256-ec-base_st-nopn_{}'.format(i) for i in range(10)]
layer_map['ec-ec-base'] = ['ec-ec-base_{}'.format(i) for i in range(10)]
layer_map['spec-ec-base'] = ['spec-ec-base_{}'.format(i) for i in range(10)]
layer_map['mel256-ec-small'] = ['mel256-ec-small_{}'.format(i) for i in range(5)]
layer_map['mel256-ec-large'] = ['mel256-ec-large_{}'.format(i) for i in [0,2,4,6,8,10,12,14,16,18,19]]
layer_map['mel256-ec-large_st-nopn'] = ['mel256-ec-large_st-nopn_{}'.format(i) for i in [0,2,4,6,8,10,12,14,16,18,19]]

layer_map['BEATs_iter1'] = ['BEATs_iter1_{}'.format(i) for i in range(13)]
layer_map['BEATs_iter2'] = ['BEATs_iter2_{}'.format(i) for i in range(13)]
layer_map['BEATs_iter3'] = ['BEATs_iter3_{}'.format(i) for i in range(13)]
layer_map['BEATs_iter3_finetuned_on_AS2M_cpt1'] = ['BEATs_iter3_finetuned_on_AS2M_cpt1_{}'.format(i) for i in range(13)]

layer_map['dasheng_base'] = ['dasheng_base_{}'.format(i) for i in range(12)]
layer_map['dasheng_base_ft-as'] = ['dasheng_base_ft-as_{}'.format(i) for i in range(12)]
layer_map['dasheng_06B'] = ['dasheng_06B_{}'.format(i) for i in [0,3,6,9,12,15,18,21,24,27,30,31]]
layer_map['dasheng_12B'] = ['dasheng_12B_{}'.format(i) for i in [0,4,8,12,16,20,24,28,32,36,39]]

layer_map['braindnn_ast'] = ['Conv2d(1, 768, kernel_size=(16, 16), stride=(10, 10))--0'] + \
                            [f'Linear(in_features=3072, out_features=768, bias=True)--{i}' for i in range(12)] + ['Final']
layer_map['braindnn_vggish'] = [f'ReLU(inplace=True)--{i}' for i in range(9)] + \
                               [f'MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)--{i}' for i in range(4)]
layer_map['braindnn_dcase2020'] = ['GRU(64, 256, batch_first=True, bidirectional=True)--0--hidden', 
                                   'GRU(512, 256, batch_first=True, bidirectional=True)--0--hidden', 
                                   'GRU(512, 256, batch_first=True, bidirectional=True)--1--hidden', 
                                   'GRU(512, 256, batch_first=True)--0--hidden', 
                                   'Linear(in_features=256, out_features=4367, bias=True)--0']
layer_map['braindnn_sepformer'] = ['Conv1d(1, 256, kernel_size=(16,), stride=(8,), bias=False)--0'] + \
                                  [f'Linear(in_features=1024, out_features=256, bias=True)--{i}' for i in range(32)]
layer_map['braindnn_metricgan'] = ['LSTM(257, 200, num_layers=2, batch_first=True, bidirectional=True)--l1--cell',
                                   'LSTM(257, 200, num_layers=2, batch_first=True, bidirectional=True)--l2--cell',
                                   'Linear(in_features=400, out_features=300, bias=True)--0',
                                   'Linear(in_features=300, out_features=257, bias=True)--0']
layer_map['braindnn_wav2vec2'] = ['Embedding', 'Encoder_1', 'Encoder_2', 'Encoder_3', 'Encoder_4', 'Encoder_5', 'Encoder_6',
                                  'Encoder_7', 'Encoder_8', 'Encoder_9', 'Encoder_10', 'Encoder_11', 'Encoder_12']
layer_map['braindnn_s2t'] = ['Embedding', 'Encoder_1', 'Encoder_2', 'Encoder_3', 'Encoder_4', 'Encoder_5', 'Encoder_6',
                                'Encoder_7', 'Encoder_8', 'Encoder_9', 'Encoder_10', 'Encoder_11', 'Encoder_12']
layer_map['braindnn_zerospeech'] = [f'ReLU(inplace=True)--{i}' for i in range(5)]
layer_map['braindnn_spectemp_filters'] = ['braindnn_spectemp_filters_avgpool']
layer_map['braindnn_deepspeech'] = ['Hardtanh(min_val=0, max_val=20)--0',
                                    'Hardtanh(min_val=0, max_val=20)--1',
                                    'LSTM(1312, 1024, bidirectional=True)--0--cell',
                                    'LSTM(1024, 1024, bidirectional=True)--0--cell',
                                    'LSTM(1024, 1024, bidirectional=True)--1--cell',
                                    'LSTM(1024, 1024, bidirectional=True)--2--cell',
                                    'LSTM(1024, 1024, bidirectional=True)--3--cell']
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

######################################################################################
m_to_invariant_key = {
        'AST': 'braindnn_ast',
        'DCASE2020': 'dcase2020',
        'DS2': 'deepspeech',
        'ResNet50multitask': 'braindnn_resnet50_multitask',
        'ResNet50audioset': 'ResNet50-Audioset',
        'ResNet50word': 'ResNet50-Word',
        'ResNet50speaker': 'ResNet50-Speaker',
        'ResNet50music': 'ResNet50-Music',
        'Kell2018multitask': 'Kell2018-Multitask',
        'Kell2018speaker': 'Kell2018-Speaker',
        'Kell2018audioset': 'Kell2018-Audioset',
        'Kell2018word': 'Kell2018-Word',
        'Kell2018music': 'Kell2018-Music', 
}

######################################################################################
downstream_scores = {
        'esc50-v2.0.0-full': 'test_top1_acc',
        'fsd50k-v1.0-full': 'test_mAP',
        'nsynth_pitch-v2.2.3-50h': 'test_pitch_acc',
        'speech_commands-v0.0.2-full': 'test_top1_acc',
        'tfds_crema_d-1.0.0-full': 'test_top1_acc',
        'tfds_gtzan-1.0.0-full': 'test_top1_acc'
}
