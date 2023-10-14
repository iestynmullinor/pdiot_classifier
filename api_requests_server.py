import requests
import json

#RUN PYTHONANYWHERE APP

DATA = [['0.008056641', '-1.0142212', '-0.0064086914', '-5.109375', '0.828125', '-2.453125'], ['-0.078125', '-0.97003174', '0.010925293', '-1.359375', '1.09375', '-1.875'], ['-0.0024414062', '-0.9824829', '-0.014465332', '-2.9375', '1.640625', '-1.265625'], ['-0.01171875', '-1.0042114', '-6.1035156e-05', '-2.453125', '-1.421875', '0.0625'], ['-0.06201172', '-1.0195923', '0.012145996', '-2.546875', '-1.375', '-0.78125'], ['-0.024169922', '-1.0061646', '0.0079956055', '-2.421875', '-0.21875', '-2.796875'], ['-0.036376953', '-0.98565674', '-0.0005493164', '-3.5625', '-1.734375', '-6.015625'], ['-0.076660156', '-1.010315', '0.023620605', '-5.375', '-4.71875', '-7.296875'], ['-0.052734375', '-0.9829712', '0.00970459', '-5.390625', '-0.234375', '-4.3125'], ['-0.036621094', '-0.93536377', '0.0060424805', '2.21875', '-0.09375', '-1.40625'], ['-0.013427734', '-0.95684814', '-0.0071411133', '5.34375', '3.015625', '3.671875'], ['-0.040039062', '-0.97857666', '-0.026184082', '7.875', '3.296875', '2.015625'], ['-0.028076172', '-0.9612427', '-0.039367676', '11.8125', '9.453125', '6.59375'], ['-0.022949219', '-0.98760986', '-0.0068969727', '7.78125', '9.140625', '11.90625'], ['-0.021240234', '-1.0176392', '0.018737793', '-1.515625', '4.234375', '-2.03125'], ['-0.08325195', '-0.9483032', '0.044128418', '3.703125', '2.640625', '1.59375'], ['-0.038085938', '-0.9644165', '0.04534912', '-1.65625', '-0.390625', '-1.9375'], ['0.0009765625', '-0.9927368', '-0.054016113', '-2.0625', '3.140625', '0.8125'], ['-0.036865234', '-1.0154419', '-0.014709473', '0.5625', '2.390625', '0.40625'], ['-0.049804688', '-1.0042114', '0.0021362305', '4.96875', '4.703125', '-1.046875'], ['-0.030029297', '-0.9741821', '-0.042785645', '5.109375', '4.34375', '-0.4375'], ['-0.03466797', '-0.9656372', '-0.07989502', '2.0', '6.328125', '-0.125'], ['0.0014648438', '-0.9885864', '0.022399902', '-1.15625', '3.546875', '-0.796875'], ['-0.091308594', '-0.98638916', '0.015075684', '-2.203125', '-0.703125', '-0.875'], ['-0.07470703', '-0.9920044', '0.008972168', '-5.625', '-4.796875', '-2.609375'], ['0.03515625', '-0.99835205', '-0.015197754', '-4.484375', '-3.953125', '-0.0625'], ['-0.05517578', '-1.0076294', '0.016784668', '-3.046875', '-2.578125', '1.03125'], ['-0.026367188', '-1.0078735', '0.00091552734', '-2.453125', '-0.15625', '-2.25'], ['-0.049072266', '-0.98687744', '0.01361084', '1.765625', '0.890625', '-2.953125'], ['-0.045898438', '-0.9902954', '-0.0010375977', '-0.78125', '1.328125', '-3.359375'], ['-0.040527344', '-0.98931885', '0.0138549805', '-4.03125', '0.59375', '-4.796875'], ['-0.029052734', '-0.99420166', '0.010681152', '-3.53125', '-0.5625', '-4.34375'], ['-0.05102539', '-0.98565674', '0.003112793', '-2.375', '0.296875', '-3.984375'], ['-0.046875', '-0.9729614', '0.008483887', '-2.703125', '0.515625', '-2.359375'], ['-0.032226562', '-0.96466064', '-0.00592041', '1.828125', '0.828125', '-0.9375'], ['-0.03540039', '-0.96710205', '-0.027893066', '3.046875', '2.453125', '1.3125'], ['-0.017822266', '-0.9541626', '-0.035949707', '7.9375', '4.328125', '2.0'], ['-0.023925781', '-0.9968872', '-0.027648926', '6.828125', '5.84375', '6.484375'], ['-0.076416016', '-0.9883423', '0.04144287', '3.640625', '1.34375', '4.484375'], ['0.045898438', '-1.0025024', '-0.0017700195', '-2.84375', '7.703125', '-0.875'], ['-0.14868164', '-0.9776001', '0.022644043', '4.53125', '0.921875', '1.5'], ['0.0048828125', '-0.99786377', '0.01361084', '-2.0625', '2.703125', '-0.84375'], ['-0.025878906', '-0.9817505', '-0.044006348', '-0.875', '1.703125', '-0.78125'], ['-0.024414062', '-0.9873657', '-0.0178833', '0.921875', '1.921875', '-1.828125'], ['-0.03149414', '-0.99468994', '-0.027648926', '1.53125', '4.234375', '-1.8125'], ['-0.043701172', '-0.9846802', '-0.034484863', '2.34375', '2.65625', '-0.28125'], ['-0.01586914', '-0.9871216', '-0.035461426', '0.515625', '0.859375', '0.953125'], ['0.008544922', '-0.991272', '0.024353027', '-3.9375', '0.234375', '-1.03125'], ['-0.11328125', '-0.9871216', '0.04168701', '-1.203125', '-1.9375', '0.484375'], ['-0.015136719', '-0.9902954', '-0.020324707', '-2.40625', '0.09375', '-0.21875'], ['-0.022216797', '-0.9846802', '-0.0022583008', '0.109375', '0.765625', '0.328125'], ['-0.059814453', '-1.005188', '0.03289795', '-1.453125', '0.234375', '-0.15625'], ['-0.03881836', '-0.9902954', '0.015075684', '-0.828125', '0.9375', '-0.703125'], ['-0.040039062', '-0.9741821', '0.0138549805', '0.953125', '0.40625', '-1.953125'], ['-0.0390625', '-0.9937134', '-0.0056762695', '3.625', '0.5', '-2.40625'], ['-0.0546875', '-1.0100708', '0.0021362305', '3.03125', '2.625', '-2.359375'], ['-0.064453125', '-0.99713135', '-0.020324707', '-1.0625', '2.328125', '-2.953125'], ['-0.022460938', '-0.97662354', '-0.016662598', '-1.25', '2.40625', '-1.53125'], ['-0.0041503906', '-0.9727173', '-0.035705566', '0.203125', '0.90625', '0.625'], ['-0.028564453', '-0.96099854', '-0.042785645', '4.234375', '1.84375', '1.921875'], ['0.005126953', '-0.9697876', '-0.066223145', '1.359375', '2.421875', '0.640625'], ['-0.009277344', '-0.96051025', '-0.06304932', '6.890625', '6.265625', '3.25'], ['-0.017333984', '-0.98931885', '-0.03765869', '1.890625', '5.3125', '6.4375'], ['-0.030273438', '-1.0449829', '-0.0012817383', '-3.65625', '0.5', '3.890625'], ['0.053222656', '-0.95196533', '-0.024963379', '-3.640625', '1.125', '-0.828125'], ['-0.08081055', '-0.9812622', '0.0045776367', '2.5', '4.296875', '1.40625'], ['0.0075683594', '-0.9717407', '0.0045776367', '-0.921875', '3.875', '-0.6875'], ['-0.007080078', '-0.9805298', '-0.046447754', '-0.140625', '4.578125', '-0.546875'], ['-0.025390625', '-1.020813', '-0.018859863', '1.359375', '3.59375', '-0.578125'], ['-0.033691406', '-0.9961548', '-0.022033691', '2.28125', '3.25', '-1.34375'], ['-0.0034179688', '-0.9614868', '-0.07647705', '1.4375', '2.0625', '-0.625'], ['-0.029541016', '-0.9780884', '0.0060424805', '-3.640625', '-1.9375', '-0.921875'], ['-0.02709961', '-0.9890747', '-0.016174316', '-4.34375', '2.625', '-1.6875'], ['-0.050048828', '-0.9841919', '0.00970459', '-0.484375', '0.8125', '-0.5625'], ['-0.025146484', '-0.98272705', '0.005554199', '-0.421875', '0.453125', '1.1875']]
json_data = json.dumps(DATA)

url = 'http://iestynmullinor.pythonanywhere.com/predict'
url_base = 'http://iestynmullinor.pythonanywhere.com/'
headers = {'Content-Type': 'application/json'}

# Make a GET request to the API
response_get= requests.get(url_base)
print(response_get.json())

# Make a POST request with JSON data
response_post = requests.post(url, data=json_data, headers=headers)

# Handle the response from the API
print(response_post.json())