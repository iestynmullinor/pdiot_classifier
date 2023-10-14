import numpy as np
from keras.models import load_model
UNIQUE_LABELS = ['misc_movements&normal_breathing', 'sitting&singing', 'standing&talking', 'sitting&normal_breathing', 'standing&laughing', 'lying_down_back&talking', 'standing&normal_breathing', 'lying_down_back&coughing', 'standing&singing', 'shuffle_walking&normal_breathing', 'descending_stairs&normal_breathing', 'sitting&eating', 'standing&coughing', 'lying_down_stomach&normal_breathing', 'lying_down_stomach&talking', 'lying_down_left&hyperventilating', 'sitting&hyperventilating', 'lying_down_back&singing', 'lying_down_right&hyperventilating', 'walking&normal_breathing', 'sitting&coughing', 'sitting&talking', 'lying_down_right&coughing', 'lying_down_stomach&hyperventilating', 'lying_down_left&normal_breathing', 'standing&hyperventilating', 'lying_down_stomach&laughing', 'lying_down_left&coughing', 'standing&eating', 'running&normal_breathing', 'lying_down_stomach&singing', 'lying_down_back&hyperventilating', 'lying_down_back&normal_breathing', 'lying_down_right&normal_breathing', 'lying_down_left&laughing', 'lying_down_left&talking', 'ascending_stairs&normal_breathing', 'lying_down_right&laughing', 'lying_down_right&singing', 'lying_down_right&talking', 'lying_down_back&laughing', 'sitting&laughing', 'lying_down_stomach&coughing', 'lying_down_left&singing']

# this is lying down left and singing for tests
LYING_LEFT_SINGING = [['-0.6437988', '-0.014953613', '-0.77716064', '-0.140625', '0.578125', '0.265625'], ['-0.6464844', '-0.0071411133', '-0.7774048', '-0.609375', '-0.453125', '0.515625'], ['-0.638916', '-0.014465332', '-0.78082275', '-1.609375', '0.171875', '0.53125'], ['-0.63134766', '-0.026428223', '-0.7605591', '-0.265625', '0.453125', '0.1875'], ['-0.6166992', '-0.0154418945', '-0.7546997', '0.21875', '0.515625', '-0.046875'], ['-0.6113281', '-0.014953613', '-0.7197876', '0.015625', '1.125', '0.765625'], ['-0.6755371', '-0.026916504', '-0.82696533', '-1.765625', '-1.5625', '0.734375'], ['-0.66625977', '-0.026672363', '-0.8245239', '0.4375', '1.515625', '-0.1875'], ['-0.61865234', '-0.011779785', '-0.7542114', '0.34375', '1.015625', '0.0'], ['-0.59936523', '-0.015197754', '-0.74053955', '0.28125', '0.765625', '0.109375'], ['-0.61987305', '-0.019104004', '-0.75128174', '-0.796875', '1.03125', '0.40625'], ['-0.6345215', '-0.016662598', '-0.7649536', '-0.59375', '1.09375', '0.5'], ['-0.6333008', '-0.023742676', '-0.7564087', '0.078125', '0.96875', '0.265625'], ['-0.6286621', '-0.026428223', '-0.7625122', '0.671875', '0.484375', '0.1875'], ['-0.6352539', '-0.023742676', '-0.7827759', '1.59375', '0.15625', '-0.109375'], ['-0.63012695', '-0.010314941', '-0.7859497', '1.28125', '-0.953125', '0.015625'], ['-0.61572266', '-0.0134887695', '-0.77008057', '0.734375', '-0.4375', '-0.140625'], ['-0.62646484', '-0.016906738', '-0.788147', '-0.984375', '-0.515625', '0.203125'], ['-0.6347656', '-0.014221191', '-0.79229736', '-1.796875', '-0.734375', '0.234375'], ['-0.6254883', '-0.022521973', '-0.7732544', '-1.3125', '0.328125', '-0.140625'], ['-0.6230469', '-0.024963379', '-0.78326416', '-0.8125', '-0.71875', '-0.3125'], ['-0.6218262', '-0.019836426', '-0.7942505', '-1.078125', '0.078125', '0.0'], ['-0.61450195', '-0.028381348', '-0.77008057', '-0.328125', '0.828125', '-0.09375'], ['-0.61376953', '-0.02130127', '-0.784729', '-0.28125', '1.609375', '-0.34375'], ['-0.6242676', '-0.023742676', '-0.79107666', '-0.03125', '1.171875', '-0.078125'], ['-0.6281738', '-0.01763916', '-0.7842407', '-0.28125', '1.96875', '-0.4375'], ['-0.6171875', '-0.022277832', '-0.7810669', '-0.3125', '2.265625', '-0.203125'], ['-0.6159668', '-0.018127441', '-0.7593384', '-0.421875', '1.453125', '-0.15625'], ['-0.6164551', '-0.01739502', '-0.767395', '-0.796875', '1.234375', '-0.109375'], ['-0.6213379', '-0.022521973', '-0.7810669', '-0.078125', '1.8125', '0.0'], ['-0.61621094', '-0.016662598', '-0.77008057', '-0.65625', '1.140625', '-0.0625'], ['-0.6291504', '-0.025939941', '-0.7871704', '-1.671875', '0.125', '0.015625'], ['-0.62060547', '-0.024230957', '-0.770813', '-1.40625', '0.34375', '0.03125'], ['-0.6176758', '-0.02154541', '-0.77178955', '-0.375', '-0.390625', '-0.1875'], ['-0.61938477', '-0.02545166', '-0.7896118', '-0.609375', '-1.328125', '0.046875'], ['-0.6135254', '-0.027160645', '-0.7781372', '-0.3125', '-0.765625', '-0.03125'], ['-0.62646484', '-0.023986816', '-0.8064575', '-0.640625', '-0.28125', '-0.109375'], ['-0.61621094', '-0.0335083', '-0.7896118', '-0.21875', '0.765625', '-0.53125'], ['-0.6086426', '-0.023498535', '-0.7962036', '-0.546875', '0.21875', '-0.359375'], ['-0.6062012', '-0.023498535', '-0.7815552', '-0.296875', '0.4375', '0.09375'], ['-0.611084', '-0.022033691', '-0.7752075', '0.109375', '1.28125', '0.046875'], ['-0.6052246', '-0.031555176', '-0.7839966', '-0.234375', '0.890625', '0.359375'], ['-0.6081543', '-0.026672363', '-0.7774048', '-0.3125', '1.015625', '0.3125'], ['-0.62109375', '-0.02545166', '-0.7888794', '-0.296875', '0.46875', '0.390625'], ['-0.6135254', '-0.020812988', '-0.8008423', '0.328125', '0.703125', '0.03125'], ['-0.6101074', '-0.023254395', '-0.7937622', '0.0', '0.34375', '0.40625'], ['-0.60253906', '-0.023498535', '-0.7737427', '0.78125', '-0.015625', '0.109375'], ['-0.6052246', '-0.019348145', '-0.7752075', '-0.390625', '0.671875', '0.25'], ['-0.609375', '-0.009094238', '-0.7905884', '-0.046875', '1.375', '0.359375'], ['-0.61987305', '-0.024719238', '-0.7966919', '-1.234375', '1.828125', '0.3125'], ['-0.61376953', '-0.024719238', '-0.7866821', '-0.171875', '1.625', '0.265625'], ['-0.62109375', '-0.02935791', '-0.78326416', '-0.59375', '0.96875', '0.125'], ['-0.6088867', '-0.02178955', '-0.79693604', '0.09375', '0.734375', '0.171875'], ['-0.607666', '-0.025939941', '-0.7769165', '-0.328125', '1.09375', '0.421875'], ['-0.61083984', '-0.016174316', '-0.78692627', '0.0625', '0.703125', '0.375'], ['-0.6142578', '-0.02130127', '-0.7937622', '0.015625', '0.734375', '0.3125'], ['-0.61206055', '-0.02545166', '-0.7920532', '0.34375', '0.765625', '0.171875'], ['-0.60668945', '-0.020812988', '-0.786438', '-0.25', '1.015625', '0.1875'], ['-0.6098633', '-0.027404785', '-0.784729', '-0.609375', '1.125', '0.265625'], ['-0.6113281', '-0.024719238', '-0.78985596', '-0.453125', '0.90625', '0.09375'], ['-0.61621094', '-0.027648926', '-0.79644775', '-1.21875', '1.265625', '0.40625'], ['-0.6027832', '-0.0154418945', '-0.79229736', '0.21875', '0.796875', '0.15625'], ['-0.6140137', '-0.026428223', '-0.77716064', '-0.546875', '1.203125', '0.203125'], ['-0.6142578', '-0.030822754', '-0.79107666', '0.0625', '1.046875', '0.28125'], ['-0.6154785', '-0.02935791', '-0.7918091', '-0.265625', '0.859375', '0.265625'], ['-0.6184082', '-0.024230957', '-0.78570557', '0.078125', '0.6875', '0.265625'], ['-0.60791016', '-0.023498535', '-0.7925415', '0.078125', '0.21875', '0.359375'], ['-0.60302734', '-0.018615723', '-0.7866821', '0.140625', '0.65625', '0.28125'], ['-0.6047363', '-0.024475098', '-0.7854614', '-0.375', '0.96875', '0.34375'], ['-0.611084', '-0.018371582', '-0.78570557', '-0.03125', '1.375', '0.34375'], ['-0.6152344', '-0.022277832', '-0.7876587', '-0.234375', '1.40625', '0.21875'], ['-0.61816406', '-0.027404785', '-0.7937622', '-0.21875', '1.28125', '0.125'], ['-0.61499023', '-0.022766113', '-0.79400635', '-0.140625', '1.140625', '0.125'], ['-0.61621094', '-0.027893066', '-0.78131104', '-0.015625', '1.03125', '0.21875'], ['-0.6057129', '-0.032043457', '-0.767395', '0.640625', '1.0625', '0.078125']]

# this is sitting and coughing for tests
SITTING_COUGHING = [['0.008056641', '-1.0142212', '-0.0064086914', '-5.109375', '0.828125', '-2.453125'], ['-0.078125', '-0.97003174', '0.010925293', '-1.359375', '1.09375', '-1.875'], ['-0.0024414062', '-0.9824829', '-0.014465332', '-2.9375', '1.640625', '-1.265625'], ['-0.01171875', '-1.0042114', '-6.1035156e-05', '-2.453125', '-1.421875', '0.0625'], ['-0.06201172', '-1.0195923', '0.012145996', '-2.546875', '-1.375', '-0.78125'], ['-0.024169922', '-1.0061646', '0.0079956055', '-2.421875', '-0.21875', '-2.796875'], ['-0.036376953', '-0.98565674', '-0.0005493164', '-3.5625', '-1.734375', '-6.015625'], ['-0.076660156', '-1.010315', '0.023620605', '-5.375', '-4.71875', '-7.296875'], ['-0.052734375', '-0.9829712', '0.00970459', '-5.390625', '-0.234375', '-4.3125'], ['-0.036621094', '-0.93536377', '0.0060424805', '2.21875', '-0.09375', '-1.40625'], ['-0.013427734', '-0.95684814', '-0.0071411133', '5.34375', '3.015625', '3.671875'], ['-0.040039062', '-0.97857666', '-0.026184082', '7.875', '3.296875', '2.015625'], ['-0.028076172', '-0.9612427', '-0.039367676', '11.8125', '9.453125', '6.59375'], ['-0.022949219', '-0.98760986', '-0.0068969727', '7.78125', '9.140625', '11.90625'], ['-0.021240234', '-1.0176392', '0.018737793', '-1.515625', '4.234375', '-2.03125'], ['-0.08325195', '-0.9483032', '0.044128418', '3.703125', '2.640625', '1.59375'], ['-0.038085938', '-0.9644165', '0.04534912', '-1.65625', '-0.390625', '-1.9375'], ['0.0009765625', '-0.9927368', '-0.054016113', '-2.0625', '3.140625', '0.8125'], ['-0.036865234', '-1.0154419', '-0.014709473', '0.5625', '2.390625', '0.40625'], ['-0.049804688', '-1.0042114', '0.0021362305', '4.96875', '4.703125', '-1.046875'], ['-0.030029297', '-0.9741821', '-0.042785645', '5.109375', '4.34375', '-0.4375'], ['-0.03466797', '-0.9656372', '-0.07989502', '2.0', '6.328125', '-0.125'], ['0.0014648438', '-0.9885864', '0.022399902', '-1.15625', '3.546875', '-0.796875'], ['-0.091308594', '-0.98638916', '0.015075684', '-2.203125', '-0.703125', '-0.875'], ['-0.07470703', '-0.9920044', '0.008972168', '-5.625', '-4.796875', '-2.609375'], ['0.03515625', '-0.99835205', '-0.015197754', '-4.484375', '-3.953125', '-0.0625'], ['-0.05517578', '-1.0076294', '0.016784668', '-3.046875', '-2.578125', '1.03125'], ['-0.026367188', '-1.0078735', '0.00091552734', '-2.453125', '-0.15625', '-2.25'], ['-0.049072266', '-0.98687744', '0.01361084', '1.765625', '0.890625', '-2.953125'], ['-0.045898438', '-0.9902954', '-0.0010375977', '-0.78125', '1.328125', '-3.359375'], ['-0.040527344', '-0.98931885', '0.0138549805', '-4.03125', '0.59375', '-4.796875'], ['-0.029052734', '-0.99420166', '0.010681152', '-3.53125', '-0.5625', '-4.34375'], ['-0.05102539', '-0.98565674', '0.003112793', '-2.375', '0.296875', '-3.984375'], ['-0.046875', '-0.9729614', '0.008483887', '-2.703125', '0.515625', '-2.359375'], ['-0.032226562', '-0.96466064', '-0.00592041', '1.828125', '0.828125', '-0.9375'], ['-0.03540039', '-0.96710205', '-0.027893066', '3.046875', '2.453125', '1.3125'], ['-0.017822266', '-0.9541626', '-0.035949707', '7.9375', '4.328125', '2.0'], ['-0.023925781', '-0.9968872', '-0.027648926', '6.828125', '5.84375', '6.484375'], ['-0.076416016', '-0.9883423', '0.04144287', '3.640625', '1.34375', '4.484375'], ['0.045898438', '-1.0025024', '-0.0017700195', '-2.84375', '7.703125', '-0.875'], ['-0.14868164', '-0.9776001', '0.022644043', '4.53125', '0.921875', '1.5'], ['0.0048828125', '-0.99786377', '0.01361084', '-2.0625', '2.703125', '-0.84375'], ['-0.025878906', '-0.9817505', '-0.044006348', '-0.875', '1.703125', '-0.78125'], ['-0.024414062', '-0.9873657', '-0.0178833', '0.921875', '1.921875', '-1.828125'], ['-0.03149414', '-0.99468994', '-0.027648926', '1.53125', '4.234375', '-1.8125'], ['-0.043701172', '-0.9846802', '-0.034484863', '2.34375', '2.65625', '-0.28125'], ['-0.01586914', '-0.9871216', '-0.035461426', '0.515625', '0.859375', '0.953125'], ['0.008544922', '-0.991272', '0.024353027', '-3.9375', '0.234375', '-1.03125'], ['-0.11328125', '-0.9871216', '0.04168701', '-1.203125', '-1.9375', '0.484375'], ['-0.015136719', '-0.9902954', '-0.020324707', '-2.40625', '0.09375', '-0.21875'], ['-0.022216797', '-0.9846802', '-0.0022583008', '0.109375', '0.765625', '0.328125'], ['-0.059814453', '-1.005188', '0.03289795', '-1.453125', '0.234375', '-0.15625'], ['-0.03881836', '-0.9902954', '0.015075684', '-0.828125', '0.9375', '-0.703125'], ['-0.040039062', '-0.9741821', '0.0138549805', '0.953125', '0.40625', '-1.953125'], ['-0.0390625', '-0.9937134', '-0.0056762695', '3.625', '0.5', '-2.40625'], ['-0.0546875', '-1.0100708', '0.0021362305', '3.03125', '2.625', '-2.359375'], ['-0.064453125', '-0.99713135', '-0.020324707', '-1.0625', '2.328125', '-2.953125'], ['-0.022460938', '-0.97662354', '-0.016662598', '-1.25', '2.40625', '-1.53125'], ['-0.0041503906', '-0.9727173', '-0.035705566', '0.203125', '0.90625', '0.625'], ['-0.028564453', '-0.96099854', '-0.042785645', '4.234375', '1.84375', '1.921875'], ['0.005126953', '-0.9697876', '-0.066223145', '1.359375', '2.421875', '0.640625'], ['-0.009277344', '-0.96051025', '-0.06304932', '6.890625', '6.265625', '3.25'], ['-0.017333984', '-0.98931885', '-0.03765869', '1.890625', '5.3125', '6.4375'], ['-0.030273438', '-1.0449829', '-0.0012817383', '-3.65625', '0.5', '3.890625'], ['0.053222656', '-0.95196533', '-0.024963379', '-3.640625', '1.125', '-0.828125'], ['-0.08081055', '-0.9812622', '0.0045776367', '2.5', '4.296875', '1.40625'], ['0.0075683594', '-0.9717407', '0.0045776367', '-0.921875', '3.875', '-0.6875'], ['-0.007080078', '-0.9805298', '-0.046447754', '-0.140625', '4.578125', '-0.546875'], ['-0.025390625', '-1.020813', '-0.018859863', '1.359375', '3.59375', '-0.578125'], ['-0.033691406', '-0.9961548', '-0.022033691', '2.28125', '3.25', '-1.34375'], ['-0.0034179688', '-0.9614868', '-0.07647705', '1.4375', '2.0625', '-0.625'], ['-0.029541016', '-0.9780884', '0.0060424805', '-3.640625', '-1.9375', '-0.921875'], ['-0.02709961', '-0.9890747', '-0.016174316', '-4.34375', '2.625', '-1.6875'], ['-0.050048828', '-0.9841919', '0.00970459', '-0.484375', '0.8125', '-0.5625'], ['-0.025146484', '-0.98272705', '0.005554199', '-0.421875', '0.453125', '1.1875']]

# this is lying on stomach breathing normally
STOMACH_NORMAL = [['0.46191406', '0.020935059', '-0.8526001', '0.125', '1.640625', '-0.640625'], ['0.45507812', '0.024841309', '-0.84820557', '0.109375', '1.546875', '-0.703125'], ['0.4543457', '0.015319824', '-0.8467407', '0.015625', '1.71875', '-0.578125'], ['0.45507812', '0.019958496', '-0.8340454', '-0.0625', '1.734375', '-0.515625'], ['0.46166992', '0.019958496', '-0.84576416', '0.078125', '1.625', '-0.671875'], ['0.45776367', '0.02142334', '-0.8513794', '0.0625', '1.671875', '-0.578125'], ['0.46264648', '0.014343262', '-0.8452759', '0.0625', '1.640625', '-0.703125'], ['0.45947266', '0.020446777', '-0.8477173', '0.1875', '1.75', '-0.59375'], ['0.46020508', '0.02166748', '-0.8491821', '0.09375', '1.765625', '-0.53125'], ['0.45898438', '0.01727295', '-0.850647', '0.125', '1.84375', '-0.578125'], ['0.46362305', '0.023376465', '-0.84576416', '0.125', '2.015625', '-0.703125'], ['0.46484375', '0.019226074', '-0.86309814', '-0.21875', '2.0625', '-0.796875'], ['0.46191406', '0.021911621', '-0.85284424', '0.140625', '1.71875', '-0.703125'], ['0.4543457', '0.027282715', '-0.85040283', '-0.125', '1.65625', '-0.65625'], ['0.45703125', '0.022644043', '-0.8452759', '0.25', '2.0625', '-0.53125'], ['0.4633789', '0.016296387', '-0.8435669', '-0.1875', '1.90625', '-0.703125'], ['0.45776367', '0.019958496', '-0.8394165', '0.109375', '1.828125', '-0.71875'], ['0.46020508', '0.016296387', '-0.84869385', '0.09375', '1.546875', '-0.671875'], ['0.46191406', '0.023376465', '-0.84503174', '0.265625', '1.78125', '-0.609375'], ['0.46313477', '0.0211792', '-0.8508911', '0.15625', '1.90625', '-0.640625'], ['0.45629883', '0.015563965', '-0.848938', '0.0', '1.890625', '-0.734375'], ['0.46069336', '0.01776123', '-0.847229', '0.0625', '2.0625', '-0.65625'], ['0.45336914', '0.020202637', '-0.847229', '-0.015625', '2.171875', '-0.734375'], ['0.45532227', '0.017028809', '-0.84820557', '0.171875', '1.921875', '-0.53125'], ['0.45874023', '0.014587402', '-0.84869385', '0.140625', '2.046875', '-0.734375'], ['0.45751953', '0.022399902', '-0.848938', '0.03125', '1.8125', '-0.828125'], ['0.45874023', '0.021911621', '-0.8501587', '-0.046875', '2.09375', '-0.546875'], ['0.4572754', '0.01751709', '-0.8496704', '0.046875', '1.765625', '-0.84375'], ['0.45996094', '0.018249512', '-0.84869385', '0.03125', '1.84375', '-0.609375'], ['0.4597168', '0.01727295', '-0.84503174', '0.203125', '2.015625', '-0.59375'], ['0.46313477', '0.01776123', '-0.8674927', '-0.421875', '2.296875', '-0.890625'], ['0.46020508', '0.018737793', '-0.8484497', '0.25', '1.8125', '-0.546875'], ['0.44995117', '0.016784668', '-0.84625244', '0.015625', '1.875', '-0.5625'], ['0.4584961', '0.022888184', '-0.85235596', '0.34375', '2.140625', '-0.59375'], ['0.45874023', '0.019714355', '-0.8530884', '-0.046875', '1.9375', '-0.640625'], ['0.45141602', '0.0211792', '-0.8567505', '0.125', '1.625', '-0.640625'], ['0.45410156', '0.0138549805', '-0.8484497', '0.0625', '1.921875', '-0.65625'], ['0.453125', '0.023132324', '-0.8521118', '0.15625', '1.703125', '-0.4375'], ['0.45629883', '0.018005371', '-0.847229', '-0.0625', '1.625', '-0.625'], ['0.45336914', '0.015319824', '-0.8452759', '0.15625', '1.640625', '-0.546875'], ['0.46313477', '0.016052246', '-0.8530884', '0.140625', '1.875', '-0.71875'], ['0.4555664', '0.020202637', '-0.8518677', '0.1875', '1.8125', '-0.5'], ['0.4633789', '0.01727295', '-0.8508911', '0.078125', '1.984375', '-0.578125'], ['0.45947266', '0.023620605', '-0.84869385', '0.046875', '1.890625', '-0.5625'], ['0.45385742', '0.016296387', '-0.8464966', '0.078125', '1.921875', '-0.578125'], ['0.45214844', '0.01776123', '-0.84625244', '0.03125', '1.796875', '-0.5'], ['0.4555664', '0.022155762', '-0.8484497', '0.03125', '1.890625', '-0.609375'], ['0.45385742', '0.015563965', '-0.84625244', '0.15625', '1.796875', '-0.578125'], ['0.45654297', '0.022644043', '-0.8518677', '0.15625', '1.8125', '-0.53125'], ['0.4567871', '0.01727295', '-0.84332275', '0.0625', '1.859375', '-0.578125'], ['0.45654297', '0.018493652', '-0.85113525', '0.015625', '1.796875', '-0.484375'], ['0.4584961', '0.018981934', '-0.8479614', '0.15625', '1.78125', '-0.5'], ['0.45410156', '0.022155762', '-0.8550415', '0.0625', '1.859375', '-0.578125'], ['0.4543457', '0.024597168', '-0.85235596', '0.03125', '1.671875', '-0.53125'], ['0.45874023', '0.027282715', '-0.8545532', '0.296875', '1.9375', '-0.390625'], ['0.45336914', '0.019714355', '-0.84991455', '0.0', '1.75', '-0.515625'], ['0.45410156', '0.02142334', '-0.84625244', '0.046875', '1.765625', '-0.515625'], ['0.45288086', '0.018737793', '-0.8479614', '0.28125', '1.953125', '-0.671875'], ['0.4555664', '0.020202637', '-0.85601807', '0.03125', '1.90625', '-0.546875'], ['0.45239258', '0.027282715', '-0.85113525', '0.0625', '1.734375', '-0.625'], ['0.45043945', '0.022399902', '-0.8508911', '0.015625', '1.5', '-0.578125'], ['0.4501953', '0.018493652', '-0.8530884', '0.1875', '1.71875', '-0.609375'], ['0.45410156', '0.022644043', '-0.8501587', '0.1875', '1.65625', '-0.640625'], ['0.45410156', '0.022155762', '-0.8526001', '0.015625', '1.71875', '-0.609375'], ['0.45776367', '0.019470215', '-0.8484497', '0.015625', '1.625', '-0.46875'], ['0.45336914', '0.017028809', '-0.84503174', '0.15625', '1.625', '-0.65625'], ['0.45239258', '0.024108887', '-0.8496704', '0.03125', '1.703125', '-0.515625'], ['0.4584961', '0.018981934', '-0.85113525', '0.078125', '1.640625', '-0.640625'], ['0.4560547', '0.018737793', '-0.8477173', '0.09375', '1.71875', '-0.625'], ['0.4543457', '0.02142334', '-0.84625244', '0.1875', '1.640625', '-0.625'], ['0.453125', '0.018005371', '-0.8521118', '-0.078125', '1.6875', '-0.65625'], ['0.44970703', '0.018493652', '-0.85040283', '-0.015625', '1.75', '-0.5'], ['0.45874023', '0.020202637', '-0.850647', '0.109375', '1.90625', '-0.640625'], ['0.45336914', '0.022155762', '-0.8442993', '0.1875', '1.578125', '-0.46875'], ['0.4555664', '0.02166748', '-0.8508911', '0.28125', '1.859375', '-0.421875']]

# this is sitting and normal breathing
SITTING_NORMAL = [['-0.1586914', '-0.9319458', '0.29437256', '-0.03125', '-1.375', '0.34375'], ['-0.15332031', '-0.9295044', '0.29144287', '-0.28125', '-1.265625', '0.3125'], ['-0.15698242', '-0.93292236', '0.291687', '0.65625', '-1.203125', '0.28125'], ['-0.15234375', '-0.92633057', '0.29559326', '-1.765625', '-1.03125', '0.0625'], ['-0.1616211', '-0.94000244', '0.29144287', '-0.03125', '-2.125', '0.421875'], ['-0.15234375', '-0.92633057', '0.29437256', '-1.59375', '-1.546875', '-0.484375'], ['-0.16748047', '-0.9368286', '0.29241943', '0.4375', '-1.703125', '0.328125'], ['-0.15600586', '-0.94000244', '0.30828857', '0.015625', '-1.265625', '0.375'], ['-0.15673828', '-0.92974854', '0.29534912', '-0.71875', '-0.9375', '-0.234375'], ['-0.15673828', '-0.9248657', '0.29364014', '-0.9375', '-1.3125', '-0.171875'], ['-0.16625977', '-0.93048096', '0.30169678', '0.109375', '-0.96875', '0.015625'], ['-0.15771484', '-0.93707275', '0.28826904', '-0.859375', '-1.515625', '0.09375'], ['-0.15698242', '-0.9302368', '0.29681396', '-0.34375', '-1.328125', '-0.0625'], ['-0.16235352', '-0.9299927', '0.29656982', '0.28125', '-0.890625', '-0.09375'], ['-0.15185547', '-0.9282837', '0.29364014', '-0.234375', '-1.5', '-0.078125'], ['-0.16113281', '-0.93341064', '0.30096436', '-0.34375', '-1.171875', '-0.0625'], ['-0.16748047', '-0.93707275', '0.30023193', '-0.4375', '-1.3125', '-0.0625'], ['-0.15454102', '-0.9307251', '0.29632568', '-0.421875', '-1.109375', '0.03125'], ['-0.16113281', '-0.9312134', '0.29144287', '-0.375', '-1.15625', '-0.109375'], ['-0.16210938', '-0.9343872', '0.3007202', '0.375', '-1.140625', '0.28125'], ['-0.15942383', '-0.9319458', '0.29266357', '0.09375', '-1.375', '0.03125'], ['-0.15820312', '-0.9302368', '0.30169678', '1.515625', '-0.796875', '0.53125'], ['-0.15112305', '-0.9402466', '0.28826904', '-1.40625', '-1.140625', '0.21875'], ['-0.15893555', '-0.92803955', '0.2921753', '0.53125', '-1.984375', '-0.28125'], ['-0.15478516', '-0.9360962', '0.29412842', '-1.90625', '-1.046875', '-0.515625'], ['-0.16625977', '-0.9365845', '0.30316162', '1.28125', '-1.5', '0.390625'], ['-0.14819336', '-0.9348755', '0.28997803', '-0.40625', '-1.09375', '0.296875'], ['-0.16259766', '-0.92755127', '0.29437256', '-0.5', '-0.46875', '-0.5'], ['-0.16894531', '-0.93048096', '0.2921753', '-0.453125', '-0.8125', '-0.03125'], ['-0.15844727', '-0.9417114', '0.2982788', '0.859375', '-1.171875', '0.15625'], ['-0.16674805', '-0.9307251', '0.29779053', '-0.390625', '-0.796875', '0.125'], ['-0.16333008', '-0.9338989', '0.2921753', '0.140625', '-0.765625', '-0.203125'], ['-0.16601562', '-0.9277954', '0.29193115', '0.140625', '-0.5', '0.21875'], ['-0.15893555', '-0.92559814', '0.291687', '-0.234375', '-0.734375', '0.015625'], ['-0.16064453', '-0.9343872', '0.28900146', '-0.25', '-0.546875', '-0.25'], ['-0.16235352', '-0.9385376', '0.29681396', '-0.765625', '-0.390625', '-0.328125'], ['-0.16088867', '-0.93414307', '0.28631592', '-1.0', '-0.640625', '-0.140625'], ['-0.15893555', '-0.9234009', '0.29388428', '-0.5625', '-0.703125', '-0.015625'], ['-0.15991211', '-0.9290161', '0.30145264', '-0.328125', '-0.125', '-0.09375'], ['-0.16601562', '-0.92926025', '0.29412842', '-0.453125', '-0.5625', '0.140625'], ['-0.15991211', '-0.9397583', '0.29656982', '-1.0625', '-0.15625', '-0.015625'], ['-0.16577148', '-0.9212036', '0.29852295', '0.4375', '-0.78125', '-0.515625'], ['-0.16357422', '-0.93829346', '0.28729248', '-2.328125', '-0.375', '-0.671875'], ['-0.16992188', '-0.9348755', '0.3078003', '-0.25', '-0.75', '0.015625'], ['-0.15917969', '-0.92926025', '0.2994995', '-0.40625', '-0.53125', '0.0625'], ['-0.15527344', '-0.9331665', '0.30145264', '-1.21875', '-0.359375', '-0.671875'], ['-0.16552734', '-0.9331665', '0.29412842', '-0.46875', '0.0625', '-0.453125'], ['-0.16308594', '-0.9307251', '0.31121826', '0.0', '-0.125', '-0.140625'], ['-0.17285156', '-0.93707275', '0.2982788', '-0.75', '-0.796875', '-0.03125'], ['-0.16113281', '-0.93048096', '0.3036499', '-0.65625', '-0.296875', '-0.453125'], ['-0.16723633', '-0.9277954', '0.30438232', '-0.21875', '-0.09375', '-0.21875'], ['-0.1652832', '-0.9295044', '0.30291748', '-0.546875', '-0.4375', '-0.03125'], ['-0.1665039', '-0.9312134', '0.30145264', '-0.875', '-0.34375', '-0.375'], ['-0.16357422', '-0.92803955', '0.31121826', '-0.78125', '-0.359375', '-0.4375'], ['-0.15991211', '-0.9273071', '0.3114624', '-0.90625', '-0.28125', '-0.421875'], ['-0.16870117', '-0.92559814', '0.30316162', '-0.875', '-0.40625', '-0.1875'], ['-0.16137695', '-0.9246216', '0.3131714', '-0.1875', '-0.65625', '-0.046875'], ['-0.16259766', '-0.9351196', '0.30828857', '-0.5', '-0.34375', '-0.171875'], ['-0.15917969', '-0.9331665', '0.30438232', '0.34375', '-0.578125', '0.15625'], ['-0.16430664', '-0.92852783', '0.31488037', '-1.6875', '-0.734375', '-0.171875'], ['-0.17285156', '-0.93341064', '0.30706787', '0.0625', '-1.140625', '0.015625'], ['-0.15893555', '-0.92193604', '0.31170654', '-1.5625', '-1.078125', '-0.796875'], ['-0.1784668', '-0.92974854', '0.31365967', '-0.40625', '-1.421875', '0.3125'], ['-0.16088867', '-0.9336548', '0.31365967', '-0.609375', '-0.78125', '0.109375'], ['-0.1640625', '-0.9273071', '0.31756592', '-0.75', '-0.296875', '-0.125'], ['-0.16894531', '-0.92755127', '0.3126831', '-0.515625', '-0.234375', '-0.125'], ['-0.16723633', '-0.92803955', '0.31732178', '0.328125', '-0.71875', '0.046875'], ['-0.1694336', '-0.9265747', '0.30804443', '-0.9375', '-0.625', '0.25'], ['-0.16455078', '-0.9229126', '0.31097412', '-0.765625', '-0.515625', '-0.578125'], ['-0.1694336', '-0.92510986', '0.3204956', '-0.765625', '-0.4375', '-0.09375'], ['-0.16479492', '-0.9204712', '0.31854248', '-0.546875', '-0.5625', '-0.34375'], ['-0.17285156', '-0.92437744', '0.3204956', '-0.75', '-0.390625', '-0.515625'], ['-0.16723633', '-0.92144775', '0.31781006', '-0.8125', '-0.5625', '-0.171875'], ['-0.16479492', '-0.9241333', '0.31732178', '-0.90625', '-0.359375', '-0.28125'], ['-0.16455078', '-0.928772', '0.32244873', '-0.84375', '-0.5', '-0.296875']]

# this is runnning and normal breathign
RUNNING_NORMAL =[['-0.34887695', '-2.0', '-0.24420166', '-50.515625', '101.359375', '23.78125'], ['-0.2421875', '-0.7276001', '0.14251709', '-28.828125', '119.640625', '15.0625'], ['-0.052978516', '0.35394287', '-0.52667236', '-14.734375', '53.8125', '4.90625'], ['1.0058594', '0.12225342', '-0.32867432', '-18.46875', '-17.671875', '14.53125'], ['-0.15112305', '0.19744873', '-0.34991455', '-0.359375', '-43.640625', '31.8125'], ['0.1743164', '0.41864014', '0.1876831', '-8.8125', '-43.140625', '28.03125'], ['-0.49145508', '0.15960693', '-0.18902588', '-16.0625', '-28.703125', '19.15625'], ['0.40283203', '-0.4602661', '-0.45147705', '-39.09375', '-76.546875', '14.203125'], ['1.999939', '-2.0', '-0.1272583', '29.15625', '18.015625', '-44.4375'], ['-1.6787109', '-2.0', '-0.27838135', '13.734375', '-20.6875', '9.453125'], ['0.3310547', '-0.7354126', '-0.237854', '22.359375', '3.859375', '-60.34375'], ['0.88842773', '-1.1792603', '0.08465576', '-20.109375', '-19.859375', '1.0625'], ['-0.16186523', '0.14105225', '-0.41827393', '-8.59375', '-12.96875', '-19.125'], ['-0.9086914', '1.7096558', '-0.27105713', '21.921875', '-15.671875', '-40.125'], ['0.42382812', '-0.37823486', '-0.18023682', '19.59375', '15.515625', '-30.546875'], ['0.2578125', '-0.2276001', '0.010192871', '6.75', '19.78125', '5.53125'], ['0.26733398', '-0.5210571', '-0.631897', '16.359375', '-0.96875', '26.53125'], ['0.2890625', '-2.0', '-0.7925415', '42.6875', '31.8125', '-36.328125'], ['-0.030517578', '-2.0', '0.1696167', '-16.390625', '70.671875', '34.03125'], ['-0.16015625', '-1.2056274', '0.22235107', '-11.328125', '115.90625', '27.28125'], ['-0.49121094', '0.2501831', '-0.93170166', '-1.75', '54.390625', '35.0625'], ['1.3156738', '0.37615967', '-0.1338501', '-25.3125', '-14.625', '5.4375'], ['0.10424805', '0.15325928', '-0.1843872', '-11.40625', '-48.953125', '49.125'], ['-0.42822266', '0.43743896', '0.0836792', '-12.390625', '-44.46875', '12.71875'], ['-0.18334961', '0.25701904', '-0.23394775', '-10.359375', '-30.953125', '14.828125'], ['0.078125', '-0.4512329', '-0.89923096', '0.40625', '-34.328125', '0.46875'], ['1.999939', '-2.0', '-1.0657349', '53.765625', '-76.609375', '-41.515625'], ['-1.5661621', '-2.0', '-0.40216064', '-22.640625', '-74.453125', '-2.9375'], ['0.5390625', '-1.7459106', '0.002380371', '46.640625', '19.546875', '-45.015625'], ['-0.08935547', '-1.125061', '-0.089904785', '-38.359375', '1.609375', '-41.890625'], ['0.44482422', '0.37249756', '-0.04473877', '-31.8125', '3.21875', '-9.796875'], ['-0.42797852', '1.1305542', '-0.2678833', '2.234375', '16.671875', '-8.703125'], ['-0.34692383', '-0.10333252', '-0.16339111', '10.0', '14.21875', '-45.96875'], ['0.61743164', '-0.2539673', '-0.10064697', '9.09375', '0.21875', '6.859375'], ['-0.3095703', '0.17327881', '0.16546631', '20.375', '39.046875', '11.828125'], ['-0.11791992', '-1.0105591', '-1.0003052', '30.71875', '8.484375', '17.046875'], ['1.999939', '-2.0', '-0.51226807', '93.078125', '52.90625', '33.15625'], ['-0.26098633', '-1.7263794', '-0.28546143', '-19.984375', '116.90625', '35.78125'], ['0.05078125', '-0.7359009', '-0.090148926', '-23.671875', '112.734375', '26.484375'], ['0.32177734', '0.2645874', '-0.27301025', '-7.734375', '33.15625', '14.109375'], ['0.24902344', '0.508728', '-0.55181885', '-8.53125', '2.03125', '18.359375'], ['0.24487305', '0.3937378', '-0.28790283', '18.546875', '-48.34375', '21.546875'], ['0.42236328', '0.16815186', '0.01751709', '14.5625', '-69.671875', '31.3125'], ['-0.24975586', '-0.066223145', '-0.677063', '-4.46875', '-77.65625', '32.703125'], ['0.4609375', '-1.4216919', '-0.67852783', '25.46875', '-8.328125', '-13.765625'], ['1.999939', '-2.0', '0.19818115', '-18.796875', '-96.828125', '8.890625'], ['-1.4934082', '-1.3901978', '-0.81915283', '-5.53125', '23.25', '-57.125'], ['1.0471191', '-0.76397705', '0.038269043', '-29.953125', '18.859375', '-34.984375'], ['0.32666016', '-0.70684814', '0.21087646', '-33.53125', '-32.828125', '-8.109375'], ['-0.5957031', '1.1112671', '-0.6170044', '-21.171875', '-17.5', '-17.0625'], ['-0.30908203', '0.38909912', '-0.10040283', '22.8125', '5.0625', '-59.71875'], ['0.57177734', '-0.41778564', '-0.32696533', '-0.890625', '14.96875', '-16.03125'], ['0.2006836', '0.12420654', '0.22088623', '6.15625', '9.40625', '14.03125'], ['0.1303711', '-0.95025635', '-1.2761841', '28.375', '-4.515625', '33.34375'], ['1.5305176', '-2.0', '-0.49835205', '38.578125', '-0.90625', '-25.703125'], ['-0.43408203', '-2.0', '-0.051330566', '-9.828125', '77.375', '12.390625'], ['0.20458984', '-1.1814575', '-0.22003174', '-26.46875', '103.328125', '5.84375'], ['-0.20385742', '0.6185913', '-0.45733643', '8.109375', '45.71875', '22.515625'], ['0.7019043', '0.2062378', '-0.22247314', '-8.828125', '-3.15625', '7.71875'], ['0.16479492', '0.29730225', '-0.269104', '-8.1875', '-31.609375', '45.203125'], ['0.040283203', '0.276062', '0.090026855', '-5.15625', '-35.671875', '16.21875'], ['-0.43554688', '0.21990967', '-0.006164551', '-16.671875', '-35.578125', '18.359375'], ['0.3774414', '-0.8508911', '-0.59161377', '-13.328125', '-36.515625', '11.546875'], ['1.999939', '-2.0', '-1.0891724', '27.671875', '-60.234375', '-20.703125'], ['-1.8034668', '-2.0', '-0.06060791', '20.859375', '-12.828125', '-6.5625'], ['0.5144043', '-0.7463989', '-0.028869629', '16.765625', '25.765625', '-50.765625'], ['0.41357422', '-0.9055786', '0.019714355', '-14.453125', '-5.390625', '2.578125'], ['-0.08666992', '0.62786865', '-0.5166626', '-16.96875', '-14.875', '-3.34375'], ['-0.26733398', '0.711853', '-0.030334473', '22.734375', '9.0625', '-35.28125'], ['0.06274414', '-0.2312622', '-0.46099854', '27.65625', '28.78125', '-26.96875'], ['0.1628418', '0.050964355', '0.1762085', '16.28125', '23.75', '0.1875'], ['0.37695312', '-0.35992432', '-1.1795044', '42.65625', '-4.21875', '23.984375'], ['0.12670898', '-2.0', '-0.8779907', '15.09375', '51.203125', '-14.796875'], ['0.0029296875', '-2.0', '0.07122803', '-13.453125', '44.375', '6.546875'], ['-0.4169922', '-1.18927', '0.32781982', '-12.5', '117.6875', '23.703125']]

# loading model
def run_test_with_1_sequence(test_sequence):
    individual_sequence_array = np.array(test_sequence, dtype=np.float32)
    individual_sequence_array = np.expand_dims(individual_sequence_array, axis=0)
    model = load_model("test_model.keras")

    # Perform inference to predict the tag
    predictions = model.predict(individual_sequence_array)

    # Post-processing to get the predicted class (index or label)
    predicted_class_index = np.argmax(predictions, axis=1)

    # Convert the predicted class index back to the tag
    predicted_tag = UNIQUE_LABELS[predicted_class_index[0]]  

    print(predicted_tag)

if __name__ == "__main__":
    run_test_with_1_sequence(LYING_LEFT_SINGING)
    print("above should be lying down left singing")

    run_test_with_1_sequence(SITTING_COUGHING)
    print("above should be sitting coughing")

    run_test_with_1_sequence(STOMACH_NORMAL)
    print("above should be lying stomach normal breathing")

    run_test_with_1_sequence(SITTING_NORMAL)
    print("above should be sitting normal breathing")

    run_test_with_1_sequence(RUNNING_NORMAL)
    print("above should be running normal breathing")