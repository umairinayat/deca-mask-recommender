import pickle

data = pickle.load(open(r'd:\Job\project_1\MICA\data\FLAME2020\FLAME_masks\FLAME_masks.pkl', 'rb'), encoding='latin1')
print('Keys:', list(data.keys()))
