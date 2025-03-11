#= Julia Video Processing & Face Tracking with Face Recognition, Emotion, Age, and Gender Detection, and Data Logging (GPU-Accelerated) =#

using OpenCV, Images, VideoIO, PyCall, CUDA, DataFrames, CSV

# Load pre-trained face detection model (Haar Cascade)
face_cascade = OpenCV.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load video file
video = VideoIO.open("video.mp4")

# Load face recognition, emotion, age, and gender detection models from Python
face_recognition = pyimport("face_recognition")
face_analysis = pyimport("deepface").DeepFace

# Move known faces database to GPU
known_faces = load("faces_database.jld2")
known_faces_gpu = CuArray(hcat(known_faces...))  # Convert to GPU tensor

# Initialize face tracking dictionary
tracked_faces = Dict{Int, Tuple{Tuple{Int,Int,Int,Int}, Array{Float32,1}}}()
next_id = 1  # ID counter for new faces

# Initialize video writer for output
output_video = VideoIO.open("output_video.mp4", "w", framerate=30)

# Initialize data storage for analysis
face_data = DataFrame(ID=Int[], Emotion=String[], Age=String[], Gender=String[], Frame=Int[])

frame_count = 0

# Read frames and detect faces
for frame in VideoIO.frames(video)
    frame_count += 1
    gray_frame = Gray.(frame)  # Convert to grayscale
    faces = OpenCV.detectMultiScale(face_cascade, gray_frame)
    
    new_tracked_faces = Dict{Int, Tuple{Tuple{Int,Int,Int,Int}, Array{Float32,1}}}()
    
    for (x, y, w, h) in faces
        face_img = frame[y:y+h, x:x+w]  # Crop face region
        
        # Convert face to numpy array (for FaceNet/Dlib)
        face_array = pycall(face_recognition.face_encodings, Any, face_img)
        
        if length(face_array) > 0
            face_vector = CuArray(face_array[1])  # Move face vector to GPU
            
            # Compare with known faces using GPU
            distances = sum((face_vector .- known_faces_gpu) .^ 2, dims=1)
            min_distance = minimum(distances)
            
            if min_distance < 0.6
                println("Face identified!")
                label = "Identified"
            else
                println("Unknown face detected!")
                label = "Unknown"
            end
            
            # Emotion, Age, and Gender detection using DeepFace
            analysis_result = pycall(face_analysis.analyze, Any, face_img, actions=["emotion", "age", "gender"])
            emotion_label = analysis_result["dominant_emotion"]
            age_label = string(analysis_result["age"])
            gender_label = analysis_result["gender"]
            
            # Face tracking logic
            assigned_id = -1
            for (id, (prev_bbox, prev_vector)) in tracked_faces
                vector_distance = sum((face_vector .- prev_vector) .^ 2)
                if vector_distance < 0.5  # Threshold for recognizing the same face in the next frame
                    assigned_id = id
                    break
                end
            end
            
            if assigned_id == -1
                assigned_id = next_id
                next_id += 1
            end
            
            new_tracked_faces[assigned_id] = ((x, y, w, h), face_vector)
            println("Tracking face ID: ", assigned_id, " | Emotion: ", emotion_label, " | Age: ", age_label, " | Gender: ", gender_label)
            
            # Store data for analysis
            push!(face_data, (assigned_id, emotion_label, age_label, gender_label, frame_count))
            
            # Draw bounding box and label
            OpenCV.rectangle!(frame, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)
            OpenCV.putText!(frame, "$label ID: $assigned_id | $emotion_label | Age: $age_label | Gender: $gender_label", (x, y-10), font=OpenCV.FONT_HERSHEY_SIMPLEX, scale=0.5, color=(255, 0, 0), thickness=2)
        end
    end
    
    # Update tracked faces dictionary
    tracked_faces = new_tracked_faces
    
    # Write frame to output video
    VideoIO.write(output_video, frame)
end

# Save collected face data to CSV for further analysis
CSV.write("face_data.csv", face_data)

# Release video file
VideoIO.close(video)
VideoIO.close(output_video)

println("Processing complete. Data saved to face_data.csv")

#= Instructions for uploading to GitHub =#
# 1. Initialize a new Git repository in the project folder:
#    git init
# 2. Add all project files:
#    git add .
# 3. Commit changes:
#    git commit -m "Initial commit - Face Recognition & Analysis Project"
# 4. Create a new repository on GitHub (manually or via CLI)
# 5. Link your local repository to GitHub:
#    git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
# 6. Push your project to GitHub:
#    git push -u origin main
# 7. Update README.md with project details and requirements
