from custom_dataset_concat import MyDataset

dataset = MyDataset('kin_hed')
print(len(dataset))

item = dataset[0]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
style = item['style']

# print(item['name'])
# print(txt)
# print(jpg.shape)
# print(hint.shape)
# print(jpg)
# print('-----------------------')
# print(hint[:,:, 0:3])

print(hint)