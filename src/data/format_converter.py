# change a .pkl dataset to a standard format
import pickle as pk
import torch
def data_converter(save_dir='data/', file_name="stamp_dataset_21_new.pkl" , label_as_strings=False, binarize_labels=False):
    with open(save_dir + file_name, "rb") as f:
        data = pk.load(f)

    #Separacion de los datos
    Train_dict = data['Train']
    Validation_dict = data['Validation']
    Test_dict = data['Test']

    #Images
    train_images = Train_dict['images']
    validation_images = Validation_dict['images']
    test_images = Test_dict['images']

    try:

        labels_train = Train_dict['labels']
        labels_val = Validation_dict['labels']
        labels_test = Test_dict['labels']

    except:

        labels_train = Train_dict['class']
        labels_val = Validation_dict['class']
        labels_test = Test_dict['class']

    class_to_label = {
            'AGN': 0.0,
            'SN': 1.0,
            'VS': 2.0,
            'asteroid': 3.0,
            'bogus': 4.0
        }
    
    if label_as_strings:
            labels_train = [class_to_label[c] for c in labels_train]
            labels_val = [class_to_label[c] for c in labels_val]
            labels_test = [class_to_label[c] for c in labels_test]
    
    train_images=train_images[(labels_train!=3)&(labels_train!=4)]
    labels_train=labels_train[(labels_train!=3)&(labels_train!=4)]
    validation_images=validation_images[(labels_val!=3)&(labels_val!=4)]
    labels_val=labels_val[(labels_val!=3)&(labels_val!=4)]
    test_images=test_images[(labels_test!=3)&(labels_test!=4)]
    labels_test=labels_test[(labels_test!=3)&(labels_test!=4)]

    #save the new pickle file
    new_data = {'Train': {'images': train_images, 'labels': labels_train},
                'Validation': {'images': validation_images, 'labels': labels_val},
                'Test': {'images': test_images, 'labels': labels_test}}
    
    if binarize_labels:
        labels_train = [1.0 if l == 1.0 else 0.0 for l in labels_train]
        labels_val = [1.0 if l == 1.0 else 0.0 for l in labels_val]
        labels_test = [1.0 if l == 1.0 else 0.0 for l in labels_test]

        new_data = {'Train': {'images': train_images, 'labels': labels_train},
                    'Validation': {'images': validation_images, 'labels': labels_val},
                    'Test': {'images': test_images, 'labels': labels_test}
                    }
            
    with open(save_dir + f"converted_binary{binarize_labels}_" + file_name, "wb") as f:
        pk.dump(new_data, f)
