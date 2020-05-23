import os

root_dir = '/root/doc3d'
train_f = open(os.path.join(root_dir, 'train.txt'), 'w')
for i in range(1, 17):
    cur_path = os.path.join(root_dir, 'img', str(i))
    names = os.listdir(cur_path)
    for name in names:
        curline = str(i) + '/' + name.split('.')[0] + '\n'
        train_f.write(name)


val_f = open(os.path.join(root_dir, 'val.txt'), 'w')
for i in range(17, 22):
    cur_path = os.path.join(root_dir, 'img', str(i))
    names = os.listdir(cur_path)
    for name in names:
        curline = str(i) + '/' + name.split('.')[0] + '\n'
        val_f.write(name)


train_f.close()
val_f.close()
