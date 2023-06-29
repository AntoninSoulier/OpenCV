import cv2

# Load the video using cv2.VideoCapture
cap = cv2.VideoCapture(0)

# Set up the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
bg_subtractor.setHistory(200)
bg_subtractor.setVarThreshold(50)

# Set up the blob detector
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

# Filter by area.
params.filterByArea = True
params.minArea = 1500

# Filter by circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

# Initialize the frame counter and the total number of people detected
frame_count = 0
people_count = 0

# Iterate through each frame of the video
while cap.isOpened():
    # Read the current frame
    success, frame = cap.read()
    if not success:
        break
    
    # Apply the background subtractor to the frame
    fg_mask = bg_subtractor.apply(frame)
    
    # Detect blobs in the foreground mask
    keypoints = detector.detect(fg_mask)
    
    # Filter the blobs to only keep the ones that are likely to be people
    people = []
    for keypoint in keypoints:
        x, y = keypoint.pt
        size = keypoint.size
        if size > 100 and size < 500:
            people.append((x, y))
    
    # Draw the blobs on the frame and display it
    for person in people:
        x, y = person
        cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1)
    cv2.imshow("Frame", frame)
    
    # Increment the frame counter and the total number of people detected
    frame_count += 1
    people_count += len(people)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When you are done processing the video, release the VideoCapture object
cap.release()
cv2.destroyAllWindows()

# Print the total number of people detected
print("Total number of people detected:", people_count)