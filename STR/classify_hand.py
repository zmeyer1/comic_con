import cv2, pickle
import numpy as np
import detect_hand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ReversePriorityQueue(object):
    # its a priority queue except lowest number comes out first
    def __init__(self):
        self.queue = []
 
    def __str__(self):
        return ' '.join([str(i) for i in self.queue])
 
    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0
 
    # for inserting an element in the queue
    def insert(self, data):
        self.queue.append(data)
 
    # for popping an element based on Priority
    def delete(self):
        try:
            min_idx = 0
            for i in range(len(self.queue)):
                if self.queue[i] < self.queue[min_idx]:
                    min_idx = i
            item = self.queue[min_idx]
            del self.queue[min_idx]
            return item
        except IndexError:
            print()
            exit()


class HandClassifier:

    def __init__(self, data_file):
        
        self.detector = detect_hand.HandDetector(num_hands=1)

        # read in data and labels from file
        with open(data_file, 'rb') as file:
            self.angles, self.labels = pickle.load(file)

        explained_variance = self.gen_initial_embeddings()
        print(f"The explained variance is: {explained_variance}, SUM: {sum(explained_variance)}")

    def gen_initial_embeddings(self, n_components = 7, display=False):
        # Performs PCA on the angle vectors

        # Standardize the data (important for PCA)
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(self.angles)

        # Create PCA object, specifying the number of components to retain
        self.pca = PCA(n_components=n_components)

        # Fit PCA to the standardized data
        self.pca.fit(scaled_data)

        # Transform the data to the new principal components
        self.embeddings = self.pca.transform(scaled_data)

        if display:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            data = {label: [] for label in self.labels}

            for i,label in enumerate(self.labels):
                data[label].append(self.embeddings[i])

            for arr in data.values():
                arr = np.array(arr)
                print(arr.shape)
                print(arr[:,0].shape)
                ax.scatter(arr[:,0], arr[:,1], arr[:,2])


            plt.show()


        # Explained variance ratio (how much variance each component explains)
        explained_variance = self.pca.explained_variance_ratio_

        return explained_variance

    def classify(self, im, is_bgr=True, display=False):
        """Takes in an image, finds two hands, converts to angle space, 
        then generates an embedding in that angle space and returns a pair of labels.
        That label is based on the k closest training data points given to the class.
        """
        label = "None"
        mean_distance = np.inf
        hand_index = {"left": 0, "right": 1}

        hands_found = self.detector.detect(im, is_bgr)
        if not hands_found.multi_hand_landmarks:
            return label, mean_distance
        for i,hand in enumerate(hands_found.multi_hand_landmarks):
            # handedness = hands_found.handedness[i][0].category_name.lower()
            angle_vec = detect_hand.generate_angle_vector(hand)
            embedding = self.pca.transform(self.scaler.transform(angle_vec.reshape(1, -1)))
            label, dists = self.get_label(embedding)
            mean_distance = np.mean(dists)

        if display:
            detect_hand.draw_landmarks_on_image(im, hands_found)
            
        return label, mean_distance
    
    def dist(self, a,b):
        # Exists as a member variable, in case descendants of this class want to change the distance function easily
        return np.sum(np.square(a-b))

    def get_label(self, embedding, k = 5):
        # uses KNN to find the label that the embedding corresponds to
        # Really CLUMSY IMPLEMENTATION
        rpq = ReversePriorityQueue()
        for i, emb in enumerate(self.embeddings):
            rpq.insert((self.dist(emb, embedding), self.labels[i]))
        neighbors = {}
        distances = []
        for _ in range(k):
            distance, label = rpq.delete()
            distances.append(distance)
            if label in neighbors.keys():
                neighbors[label] += 1
            else:
                neighbors[label] = 1
        max_count = 0
        label = None
        for k,v in neighbors.items():
            if v > max_count:
                max_count = v
                label = k
        return label, distances


def reconstuct_skeleton(angles, segments=detect_hand.hand_segments):
    # plots a reconstructed skeleton from the given angles, assumes unit length for all segments
    
    #define the origin as the right shoulder
    positions = {
        0: np.zeros(3)
    }

    segment_positions = []

    for i,segment in enumerate(segments):
        initial_pos = positions[segment[0]]
        # Two angles per segment
        angle_xy, angle_z = angles[i*2:i*2+2]
        final_pos = initial_pos + np.array((np.cos(angle_xy), np.sin(angle_xy), np.sin(angle_z)))
        positions[segment[1]] = final_pos
        # draw the segments from previous position to new position
        segment_positions.append([initial_pos, final_pos])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Create the Line3DCollection
    line_collection = Line3DCollection(segment_positions, colors='r', linewidths=2)

    # Add the collection to the plot
    ax.add_collection3d(line_collection)
    ax.autoscale()
    plt.show()



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    classifier = HandClassifier("hand_signs.pickle")

    while cap.isOpened():
        ret, frame = cap.read()

        # Check if frame was read correctly
        if not ret:
            break

        # Remove the color conversions
        # The frame is already in BGR format from cap.read()
        
        # The classify method expects BGR, which is the default from the camera
        label, dist = classifier.classify(frame, display=True)

        # The rest of your code
        if dist > 10:
            label = "None"

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"{label} ({np.round(dist,2)})", (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('window', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
