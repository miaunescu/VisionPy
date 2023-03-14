# This script will detect faces via your webcam.
# Tested with OpenCV3
from datetime import datetime

import cv2


def transparent_overlay(src, overlay, pos=(0, 0), scaleX=1, scaleY=1):
    """
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Resultant Image
    """
    overlay = cv2.resize(overlay, (0, 0), fx=scaleX, fy=scaleY)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image

    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = (
                alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
            )
    return src


def calculate_hat_position(face_position, hat_size):
    # hat should be on top of face, centered with the face

    (x, y, w, h) = face_position
    (hat_w, hat_h, hat_pos) = hat_size
    scale_factor = w * 1.5 / hat_w
    hat_w = int(hat_w * scale_factor)
    hat_h = int(hat_h * scale_factor)
    pos_hat = (x + w // 2 - hat_w // 2, y - hat_h + hat_pos + int(h * 0.15))

    print(face_position)
    print(hat_size)
    return (pos_hat, scale_factor, scale_factor)


def calculate_uniform_position(face_position, uniform_size):
    # uniforms should be under the face but not totally
    pass
    (x, y, w, h) = face_position
    (uniform_w, uniform_h, uniform_o, uniform_of, uniform_y) = uniform_size
    scale_factor = w / (uniform_o * 1.4)
    uniform_w = int(uniform_w * scale_factor)
    if uniform_y == 0:
        pos_uniform = (x + w // 2 - uniform_w // 2 - uniform_of, y + h + int(h / 12))
    else:
        pos_uniform = (x + w // 2 - uniform_w // 2 - uniform_of, y + h + uniform_y)

    return pos_uniform, scale_factor, scale_factor


def calculate_eyes_position(eyes, size_glasses):
    # hat should be on top of face, centered with the face
    #
    glassesX = eyes[0][0]
    glasses2X = eyes[1][0]
    glassesY = eyes[0][1]
    glassesW = eyes[0][2]
    glassesH = eyes[0][3]
    if eyes[1][0] < glassesX:
        glassesX = eyes[1][0]
        glasses2X = eyes[0][0]
        glassesY = eyes[1][1]
        glassesW = eyes[1][2]
        glassesH = eyes[1][3]

    print("glasses")
    middleX = glassesX + glassesW + int((glasses2X - (glassesX + glassesW)) / 2)
    scalling_fact = glassesH / size_glasses[1]
    glassesX = middleX - int((size_glasses[0] / 2) * scalling_fact)
    print(glassesX, glasses2X, glassesY, glassesH, glassesW)
    print(middleX, scalling_fact, glassesW, glassesX)
    return (glassesX, glassesY), glassesH / size_glasses[1], glassesH / size_glasses[1]


def set_initial_state():
    uniform_status = 2
    hat_status = 4


# init state
size_glasses = (754, 242)
pos_glasses = (0, 0)
size_hat = (196, 161, 0)
size_hat2 = (921, 650, 0)
size_hat3 = (1000, 856, 0)
size_hat4 = (1024, 859, 20)
pos_hat = (0, 0)
size_uniform = (766, 397, 278, -10, -10)
size_uniform2 = (1000, 601, 268, 15, -15)
size_uniform3 = (503, 561, 124, 60, -30)
pos_uniform = (0, 0)
uniform_status = 2
hat_status = 4
glasses_status = 1

set_initial_state()

# start the camera
# Change cv2.VideoCapture(1) to cv2.VideoCapture(0). It is a macOS bug.
cap = cv2.VideoCapture(0)

# Create the haar cascade1
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# frontal_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
eyesCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

profileCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)
handCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)
# palmCascade = cv2.CascadeClassifier(r"palm.xml")

overlayImage = cv2.imread(
    "images/harry_theme/glasses/glasses4.png", cv2.IMREAD_UNCHANGED
)
# hatImage = cv2.imread("hat3.png", cv2.IMREAD_UNCHANGED)
# hatImage2 = cv2.imread("hat22.png", cv2.IMREAD_UNCHANGED)
# hatImage3 = cv2.imread("hat5.png", cv2.IMREAD_UNCHANGED)
# hatImage4 = cv2.imread("hathat.png", cv2.IMREAD_UNCHANGED)
# uniformImage = cv2.imread("Harry3.png", cv2.IMREAD_UNCHANGED)
# uniformImage2 = cv2.imread("uniform2.png", cv2.IMREAD_UNCHANGED)
# uniformImage3 = cv2.imread("uniform3.png", cv2.IMREAD_UNCHANGED)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        # flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    # detect eyes
    eyes = eyesCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
        # flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    # detect profile
    profile = profileCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        # flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))
    print("Found {0} eyes!".format(len(eyes)))
    print("Found {0} profiles!".format(len(profile)))

    # Draw a rectangle around the faces
    for x, y, w, h in faces:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_area = (x, y, w, h)
        print("face location:" + str(x) + "," + str(y) + "," + str(w) + "," + str(h))
        print(
            "hat location:"
            + str(x + w // 2)
            + ","
            + str(y)
            + ","
            + str(h * 1.5 / 200)
            + ","
            + str(w * 1.5 / 200)
        )

        # add hat
        # if hat_status == 1:
        #     pos_hat, scale_x, scale_y = calculate_hat_position((x, y, w, h), size_hat)
        #     frame = transparent_overlay(frame, hatImage, pos_hat, scale_x, scale_y)
        #
        # if hat_status == 2:
        #     pos_hat, scale_x, scale_y = calculate_hat_position((x, y, w, h), size_hat2)
        #     frame = transparent_overlay(frame, hatImage2, pos_hat, scale_x, scale_y)
        #
        # if hat_status == 3:
        #     pos_hat, scale_x, scale_y = calculate_hat_position((x, y, w, h), size_hat3)
        #     frame = transparent_overlay(frame, hatImage3, pos_hat, scale_x, scale_y)
        #
        # if hat_status == 4:
        #     pos_hat, scale_x, scale_y = calculate_hat_position((x, y, w, h), size_hat4)
        #     frame = transparent_overlay(frame, hatImage4, pos_hat, scale_x, scale_y)

        # # add uniforms
        # if uniform_status == 1:
        #     pos_uniform, scale_x, scale_y = calculate_uniform_position((x, y, w, h), size_uniform)
        #     frame = transparent_overlay(frame, uniformImage, pos_uniform, scale_x, scale_y)
        #
        # if uniform_status == 2:
        #     pos_uniform, scale_x, scale_y = calculate_uniform_position((x, y, w, h), size_uniform2)
        #     frame = transparent_overlay(frame, uniformImage2, pos_uniform, scale_x, scale_y)
        #
        # if uniform_status == 3:
        #     pos_uniform, scale_x, scale_y = calculate_uniform_position((x, y, w, h), size_uniform3)
        #     frame = transparent_overlay(frame, uniformImage3, pos_uniform, scale_x, scale_y)

    # Draw a rectangle around the profile
    for x, y, w, h in profile:
        pass
    # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    # roi_area = (x,y,w,h)

    # Draw a rectangle around the palm
    # 	for (x, y, w, h) in palm:
    # 		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
    # 		roi_area = (x,y,w,h)

    # Draw a circle around the eyes
    for x, y, w, h in eyes:
        pass
    # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
    # 	cv2.circle(frame, (int(x+w/2), int(y+h/2)), int(w/2), (0, 0, 0), 2)
    # try to add real glasses

    if glasses_status == 1 and len(eyes) == 2:
        (glassesX, glassesY), scale_x, scale_y = calculate_eyes_position(
            eyes, size_glasses
        )
        frame = transparent_overlay(
            frame, overlayImage, (glassesX, glassesY), scale_x, scale_y
        )

    # Display the resulting frame
    cv2.imshow("frame", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    # 	break
    k = cv2.waitKey(1)

    if k == ord("q"):
        print("quiting")
        break
    if k == ord("p"):
        print("picture!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        roi = frame[
            roi_area[1] : roi_area[1] + roi_area[3],
            roi_area[0] : roi_area[0] + roi_area[2],
        ]
        cv2.imwrite("roi.png", roi)
    if k == ord("r"):
        print("full picture!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        name = "./images/full_picture_{0}.png".format(
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        print(name)
        cv2.imwrite(name, frame)

    if k == ord("u"):
        uniform_status = uniform_status + 1
        uniform_status = uniform_status % 4

    if k == ord("h"):
        hat_status = hat_status + 1
        hat_status = hat_status % 5

    if k == ord("g"):
        glasses_status = glasses_status + 1
        glasses_status = glasses_status % 2
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
