import os

testing_path = "/home/boyuan/ZKP/TinyQuantCNN/numpyInferenceEngine/LeNet_ORL/orl_face_dataset/testing"
training_path = "/home/boyuan/ZKP/TinyQuantCNN/numpyInferenceEngine/LeNet_ORL/orl_face_dataset/training"

testing_persons = os.listdir(testing_path)
training_persons = os.listdir(training_path)

for person in testing_persons:
    path = testing_path + '/' + person
    files = os.listdir(path)
    for file in files:
        if file[-1] == 'm':
            os.system('rm '+path+'/'+file)
            # print(path+file)

for person in training_persons:
    path = training_path + '/' + person
    files = os.listdir(path)
    for file in files:
        if file[-1] == 'm':
            os.system('rm '+path+'/'+file)
            # print(path+file)




