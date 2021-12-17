from typing import NewType
import cv2
import numpy as np
import math
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_selfie_segmentation = mp.solutions.selfie_segmentation


BG_COLOR = (192, 192, 192) # gray

def fond(image):
    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1) as selfie_segmentation:

        bg_image = cv2.imread('fond.jpg')
        bg_image = cv2.resize(bg_image,(image.shape[1],image.shape[0])) 
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = selfie_segmentation.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack(
            (results.segmentation_mask,) * 3, axis=-1) > 0.1
        # The background can be customized.
        #   a) Load an image (with the same width and height of the input image) to
        #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
        #   b) Blur the input image by applying image filtering, e.g.,
        #      bg_image = cv2.GaussianBlur(image,(55,55),0)
        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
        output_image = np.where(condition, image, bg_image)
    return output_image



def cartoonify(image):
    tublur = cv2.medianBlur(image, 29)

    # We'll cover Canny edge detection and dilation shortly
    edge = cv2.Canny(tublur, 10, 150)
    kernel = np.ones((5,5), np.uint8)
    edge = cv2.dilate(edge, kernel, iterations = 1)
    tublur[edge==255] = 0
    return tublur

def sidebyside(imageleft,imageright):
    newimheight = max(imageleft.shape[0],imageright.shape[0])
    newimwidth = imageleft.shape[1] + imageright.shape[1]
    newim = np.zeros((newimheight,newimwidth,3), np.uint8)
    newim[0:imageleft.shape[0],0:imageleft.shape[1]] = imageleft
    newim[0:imageright.shape[0],imageleft.shape[1]:] = imageright
    return newim

def filtre_gray(image):
    img_2 = image.copy()
    img_2[:,:,0] = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_2[:,:,1] = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_2[:,:,2] = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return img_2

def sepia(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_gray = np.array(gray, np.float32)/255
    #solid color
    sepia = np.ones(image.shape)
    sepia[:,:,0] *= 255 #B
    sepia[:,:,1] *= 204 #G
    sepia[:,:,2] *= 153#R
    #hadamard 
    sepia[:,:,0] *= normalized_gray #B
    sepia[:,:,1] *= normalized_gray #G
    sepia[:,:,2] *= normalized_gray #R
    return np.array(sepia, np.uint8)

def effet_miroir(image):
    flipVertical = cv2.flip(image, 1)

    return sidebyside(flipVertical[0:flipVertical[0].shape[0],0:flipVertical.shape[1]//2],image[0:image[0].shape[0],image.shape[1]//2:])

def blurring_face(img):
    # ksize
    ksize = (25, 25)
    img_blur = img.copy()
    
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:    
            for face in results.multi_face_landmarks :
                max_x = int(max([i.x for i in face.landmark]) * img.shape[1])
                min_x = int(min([i.x for i in face.landmark]) * img.shape[1])

                max_y = int(max([i.y for i in face.landmark]) * img.shape[0])
                min_y = int(min([i.y for i in face.landmark]) * img.shape[0])

                if max_x < img.shape[1] and min_x > 0 and max_y < img.shape[0] and min_y > 0:

                    #img_blur[min_y:max_y,min_x:max_x,:] = cv2.blur(img_blur[min_y:max_y,min_x:max_x,:], ksize, cv2.BORDER_DEFAULT)
                    img_blur[min_y:max_y,min_x:max_x,:] = cv2.GaussianBlur(img_blur[min_y:max_y,min_x:max_x,:], ksize , 50)
                
    return img_blur 

def face_swap(img):
    # ksize
    ksize = (25, 25)
    #img = cv2.imread(img)
    img_blur = img.copy()

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            if len(results.multi_face_landmarks) >= 2:
                f1_max_x = int(max([i.x for i in results.multi_face_landmarks[0].landmark]) * img.shape[1])
                f1_min_x = int(min([i.x for i in results.multi_face_landmarks[0].landmark]) * img.shape[1])

                f1_max_y = int(max([i.y for i in results.multi_face_landmarks[0].landmark]) * img.shape[0])
                f1_min_y = int(min([i.y for i in results.multi_face_landmarks[0].landmark]) * img.shape[0])

                f2_max_x = int(max([i.x for i in results.multi_face_landmarks[1].landmark]) * img.shape[1])
                f2_min_x = int(min([i.x for i in results.multi_face_landmarks[1].landmark]) * img.shape[1])

                f2_max_y = int(max([i.y for i in results.multi_face_landmarks[1].landmark]) * img.shape[0])
                f2_min_y = int(min([i.y for i in results.multi_face_landmarks[1].landmark]) * img.shape[0])
                
                #plt.imshow(img_blur[f1_min_y:f1_max_y,f1_min_x:f1_max_x,:])
                if f1_max_x < img.shape[1] and f1_min_x > 0 and f1_max_y < img.shape[0] and f1_min_y > 0 and f2_max_x < img.shape[1] and f2_min_x > 0 and f2_max_y < img.shape[0] and f2_min_y > 0:
                    face1 = img_blur[f1_min_y:f1_max_y,f1_min_x:f1_max_x,:].copy()
                    face1_bis = img_blur[f1_min_y:f1_max_y,f1_min_x:f1_max_x,:].copy()
                    face2 = img_blur[f2_min_y:f2_max_y,f2_min_x:f2_max_x,:].copy()
                    
                    face1 = cv2.resize(face2, (face1.shape[1],face1.shape[0])).copy()
                    face2 = cv2.resize(face1_bis, (face2.shape[1],face2.shape[0]))
                    
                    img_blur[f1_min_y:f1_max_y,f1_min_x:f1_max_x,:] = face1
                    img_blur[f2_min_y:f2_max_y,f2_min_x:f2_max_x,:] = face2
                
        return img_blur
                

def ajout_texte(image, filtre):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = image.copy()
    return cv2.putText(img,filtre,(30,175), font, 2, (255, 0, 255), 2, cv2.LINE_AA)

def appel_filtre(image, filtre):
    if filtre == 'none':
        return ajout_texte(image,filtre)
    if filtre == 'gray':
        return ajout_texte(filtre_gray(image),filtre)
    elif filtre == 'miroir':
        return ajout_texte(effet_miroir(image),filtre)
    elif filtre == 'sepia':
        return ajout_texte(sepia(image),filtre)
    elif filtre == 'ears':
        return ajout_texte(lens_filter(image,'./doggy_ears.png'),filtre)
    elif filtre == 'blurring':
        return ajout_texte(blurring_face(image),filtre)
    elif filtre == 'face landmarks':
        return ajout_texte(draw_face_landmarks(image),filtre)
    elif filtre == 'shapening':
        return ajout_texte(shapening(image),filtre)
    elif filtre == 'upSideDown':
        return ajout_texte(upSideDown(image),filtre)
    elif filtre == 'cartoonify':
        return ajout_texte(cartoonify(image),filtre)
    elif filtre == 'face_swap':
        return ajout_texte(face_swap(image),filtre)
    elif filtre == 'fond':
        return ajout_texte(fond(image),filtre)

def quadrillage(image,filtres):
    newimheight = image.shape[0]*3
    newimwidth = image.shape[1]*3
    newim = np.zeros((newimheight,newimwidth,3), np.uint8)
    height = image.shape[0]
    width = image.shape[1]    
    
    for i in range(9):
        newim[(i//3)*height:(i//3+1)*height, (i%3)*width:(i%3+1)*width] = appel_filtre(image, filtres[i])

    return newim


def get_face_landmarks(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img)  
    return results

def upSideDown(img):
    return cv2.flip(img, 0)
    

def compute_angle(point1, point2):  
    x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
    angle = -180/math.pi * math.atan(float(y2-y1)/float(x2-x1))
    
    return angle

def blend_img_with_overlay(img, overlay_img, blending_pos_x, blending_pos_y):
    img_h, img_w = img.shape[:2]
    over_h, over_w = overlay_img.shape[:2]


    crop_left = 0
    crop_right = 0
    crop_top = 0
    crop_bottom = 0
    if blending_pos_y < 0:
        crop_left = -blending_pos_y
    if blending_pos_y + over_w > img_w :
        crop_right = blending_pos_y + over_w - img_w

    if blending_pos_x < 0:
        crop_bottom = -blending_pos_x
    if blending_pos_x + over_h > img_h :
        crop_top = blending_pos_x + over_h - img_h

    new_img = img.copy()

    pos_x2 = blending_pos_x + over_h
    pos_y2 = blending_pos_y + over_w
    if crop_left < over_w and crop_right < over_w and crop_top < over_h and crop_bottom < over_h :    
        extOverlay = np.zeros(img.shape, np.uint8)
        #extOverlay[blending_pos_x:pos_x2, blending_pos_y+crop_left:pos_y2-crop_right] = overlay_img[:, crop_left:over_w-crop_right, :3] 
        extOverlay[blending_pos_x+crop_bottom:pos_x2+crop_top, blending_pos_y+crop_left:pos_y2-crop_right] = overlay_img[crop_bottom:over_h-crop_top, crop_left:over_w-crop_right, :3] 

        #cv2.imshow('test mask2', extOverlay)
    
        new_img[extOverlay>0] = extOverlay[extOverlay>0]

    
    return new_img


def lens_filter(img, png_fname):
    results = get_face_landmarks(img)
    #IMREAD_UNCHANGED = permet de garder la 4ème couche
    doggy_ears = cv2.imread(png_fname, cv2.IMREAD_UNCHANGED)
    
    new_img = img.copy()
    if results.multi_face_landmarks:
        img_h, img_w = img.shape[:2]
    
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        dog_h, dog_w = doggy_ears.shape[:2]
        face_pin_1 = face_landmarks[332]
        face_pin_2 = face_landmarks[103]
        
        angle = compute_angle((face_pin_1.x,face_pin_1.y),(face_pin_2.x,face_pin_2.y))
        
        M = cv2.getRotationMatrix2D((dog_w/2, dog_h/2), angle, 1)
        doggy_ears = cv2.warpAffine(doggy_ears, M, (dog_w, dog_h))
        
        face_right = face_landmarks[454]
        face_left = face_landmarks[234]

        face_bottom = face_landmarks[152]
        face_top = face_landmarks[10]

        
        face_w = math.sqrt((face_right.x - face_left.x)**2 + (face_right.y - face_left.y)**2)
        face_h = math.sqrt((face_top.x - face_bottom.x)**2 + (face_top.y - face_bottom.y)**2)
        
        
        ratio_w = (face_w*img_w) / dog_w
        ratio_h = (face_h*img_h) / dog_h
        
        doggy_ears = cv2.resize(doggy_ears, (int(ratio_w*dog_w), int(ratio_w*dog_h)))

        dog_h, dog_w = doggy_ears.shape[:2]

        pos_x = int(img_h * face_top.y - dog_h/2)
        pos_y = int(img_w * face_top.x - dog_w/2)

    
        new_img = blend_img_with_overlay(img, doggy_ears, pos_x, pos_y)
        
    return new_img


def shapening(img):
    kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
    img_sharpen = cv2.filter2D(img, -1, kernel)
    return img_sharpen

def draw_face_landmarks(img):
    results = get_face_landmarks(img)
    new_img = img.copy()
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image = new_img, 
                                      landmark_list = face_landmarks, 
                                      connections = mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec = None, 
                                      connection_drawing_spec = mp_drawing_styles
                                      .get_default_face_mesh_tesselation_style())
    return new_img

cam = cv2.VideoCapture(0)
#shapening
filtres = ['miroir','gray','cartoonify','face_swap','fond','blurring','face landmarks','ears','upSideDown']

with mp_face_mesh.FaceMesh(max_num_faces = 1,
                          min_detection_confidence = 0.5,
                          min_tracking_confidence = 0.5) as face_mesh:
    

    while cam.isOpened():
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        
        cv2.namedWindow('Webcam', cv2.WINDOW_GUI_NORMAL)

        cv2.imshow('Webcam', quadrillage(frame,filtres))
        #cv2.imshow('Webcam',face_swap(frame))
        #cv2.imshow('webcam',frame)
        #cv2.imshow('Doggy ears', lens_filter(frame,'./doggy_ears.png'))
        #cv2.imshow('Sharpen',shapening(frame))  
        #cv2.imshow('Face Landmarks',draw_face_landmarks(frame))    

        
        #27 = Echap
        if cv2.waitKey(1) == 27 :
            break

    cam.release()
    cv2.destroyAllWindows()
