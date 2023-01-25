import dlib
import numpy as np

selfies = os.listdir('input/selfy/')
group_pics = os.listdir('input/group_pics/')

print(f"available selfies {selfies}")
print(f"available group pics {group_pics}")
# Load the pre-trained facial recognition model
face_detector = dlib.get_frontal_face_detector()
face_embedder = dlib.face_recognition_model_v1("src/dlib_face_recognition_resnet_model_v1.dat")

# Load the selfi image of the person
selfi_image = dlib.load_rgb_image(f"input/selfy/{selfies[0]}")

# Load the group image
group_image = dlib.load_rgb_image(f"input/group_pics/{group_pics[0]}")

# Detect the face in the selfi image
selfi_faces = face_detector(selfi_image, 1)

# Extract the feature vector from the selfi face
selfi_face_descriptor = np.array(face_embedder.compute_face_descriptor(selfi_image, selfi_faces[0], 1))

# Detect faces in the group image
group_faces = face_detector(group_image, 1)

# Extract feature vectors from the group faces
group_face_descriptors = [np.array(face_embedder.compute_face_descriptor(group_image, face_pose, 1)) for face_pose in group_faces]

# Compare the selfi feature vector with the group feature vectors using cosine similarity
threshold = 0.6 # threshold for similarity
for group_descriptor in group_face_descriptors:
    similarity = np.dot(selfi_face_descriptor, group_descriptor) / (np.linalg.norm(selfi_face_descriptor) * np.linalg.norm(group_descriptor))
    if similarity > threshold:
        print("Person is in the group picture with similarity: ", similarity)
        break
else:
    print("Person is not in the group picture")
