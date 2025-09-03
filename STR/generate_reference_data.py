# Iterates through all the data and saves the angle vectors with their labels
import pickle, os, cv2
import detect_hand


def create_dataset(folder_path):
    # Label is derived from the parent folder of the dataset
    all_angles = []
    all_labels = []
    detector = detect_hand.HandDetector()
    for file in os.listdir(folder_path):
        if file.endswith('.jpeg'):
            
            im = cv2.imread(os.path.join(folder_path, file))
            hands = detector.detect(im)
            if not hands.multi_hand_landmarks :
                continue
            angles = detect_hand.generate_angle_vector(hands.multi_hand_landmarks[0]) # dataset has only since hand images
            all_angles.append(angles)
            all_labels.append(os.path.split(folder_path)[-1])
        elif os.path.isdir(os.path.join(folder_path,file)):
            angles, labels = create_dataset(os.path.join(folder_path, file))
            all_angles.extend(angles)
            all_labels.extend(labels)

    return all_angles, all_labels


if __name__ == "__main__":

    foldername = 'hand_signs'
    all_angles, all_labels = create_dataset(foldername)
    with open(f'{foldername}.pickle', 'wb') as file:
        pickle.dump((all_angles, all_labels), file, protocol=pickle.HIGHEST_PROTOCOL)
    